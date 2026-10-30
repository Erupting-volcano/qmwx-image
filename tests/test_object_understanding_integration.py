"""Synthetic protocol/storage fixtures; these tests do not measure VLM accuracy."""

from dataclasses import replace
from io import BytesIO
import json
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from app.main import create_app
from app.models.object_profile import ComponentInfo, MaterialCandidate
from app.models.object_understanding import IMAGE_TASK_IDS, ObjectUnderstandingData
from app.models.processing_result import ProcessingResult
from app.modules.base import ProcessingContext
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.modules.object_understanding_processor import ObjectUnderstandingProcessor
from app.modules.recognition_backend import RecognitionCapabilities, RecognitionResult
from app.modules.recognition_factory import RecognitionRegistration, RecognitionRegistry, RecognitionRouter
from app.modules.segmentation_backend import MaskPrediction, SegmentationCapabilities
from app.services.image_collection import ImageCollectionService, ImageInput
from app.services.profile_assembler import ProfileAssembler
from app.storage.local_storage import ObjectTransaction, storage_io_path


def sourced(asset, **fields):
    return {"asset_ids": [str(asset)], "evidence": "仅用于接口测试的可见证据", **fields}


def outputs(asset):
    return {
        "classification": {"category_candidates": [sourced(asset, category="杯子", status="candidate")]},
        "appearance": {"appearance": {"colors": [sourced(asset, text="白色", role="primary", status="observed")],
                                      "shape_description": sourced(asset, text="可见圆形开口", status="observed")}},
        "material": {"material_candidates": [sourced(asset, material_name="陶瓷", status="candidate", confidence=0.95)]},
        "components": {"visible_components": [sourced(asset, component_name="杯身", description="图片中可见的外表面", status="observed")]},
        "viewpoint": {"viewpoint": sourced(asset, viewpoint="perspective", status="candidate")},
    }


class TestOnlyRecognitionBackend:
    __test__ = False
    model_id = "test_vlm"
    model_version = "test-fixture"
    capabilities = RecognitionCapabilities(image_input=True, multi_image_input=True, structured_output=True)

    def __init__(self):
        self.calls, self.fail_assets = [], set()
        self.bad_material = self.foreign_source = self.fail_fusion = False
        self.many_unknowns = False

    def analyze(self, request):
        self.calls.append(request)
        if request.task_id == "multi_view_fusion":
            if self.fail_fusion:
                raise RuntimeError("PRIVATE_KEY PRIVATE_BASE64_RESPONSE")
            sources = [item["asset_id"] for item in request.context["image_observations"]]
            data = {"object_category": {**sourced(sources[0], category="杯子", status="candidate"), "asset_ids": sources},
                    "material_candidates": [{**sourced(sources[0], material_name="陶瓷", status="candidate"), "asset_ids": sources}],
                    "components": [{**sourced(sources[0], component_name="杯身", description="各图中可见", status="observed"), "asset_ids": sources}],
                    "cross_view_consensus": [{**sourced(sources[0], text="均可见杯状外形", status="observed"), "asset_ids": sources}],
                    "conflicting_observations": [{"field": "表面颜色", "description": "光照可能影响颜色，需要复查",
                        "claims": [sourced(sources[0], text="偏白", status="observed"), sourced(sources[1], text="偏灰", status="observed")]}]}
        else:
            asset = request.context["asset_ids"][0]
            if asset in self.fail_assets:
                raise RuntimeError("PRIVATE_KEY PRIVATE_IMAGE_DATA")
            data = {task: outputs(asset)[task] for task in request.context["task_ids"]}
            if "material" in data and self.bad_material:
                data["material"]["material_candidates"][0]["status"] = "confirmed"
            if "material" in data and self.foreign_source:
                data["material"]["material_candidates"][0]["asset_ids"] = [str(uuid4())]
            if self.many_unknowns:
                for task in data:
                    data[task]["unknown_information"] = [f"{task} 未知事项 {i}" for i in range(100)]
        return RecognitionResult(data, self.model_id, self.model_version, 0.01,
                                 metadata={self.model_id: {"fixture": True}})


def setup_service(settings, backend=None, *, task_models=None, second_backend=None, processors=None, **overrides):
    backend = backend or TestOnlyRecognitionBackend()
    settings = replace(settings, recognition_enabled=True, recognition_model=backend.model_id,
                       recognition_task_models=task_models or {}, **overrides)
    registry = RecognitionRegistry()
    for item in (backend, second_backend):
        if item is not None:
            registry.register(RecognitionRegistration(item.model_id, "Test only", item.model_id, item.model_version,
                item.capabilities, "local", builder=lambda settings, model=item: model))
    processor = ObjectUnderstandingProcessor(settings, router=RecognitionRouter(settings, registry))
    service = ImageCollectionService(settings, processors=[*(processors or []), processor])
    return service, processor, backend, settings


def collect(service, png_bytes, count=1):
    return service.collect("用户提供的名称仅作为提示", [ImageInput(f"image-{i}.png", BytesIO(png_bytes)) for i in range(count)])


def data_of(profile):
    result = next(item for item in reversed(profile.processing_results) if item.module_name == "object_understanding")
    return ObjectUnderstandingData.model_validate(result.data), result


def rerun_context(profile, settings):
    return ProcessingContext(images=tuple(profile.images), data_dir=settings.data_dir, profile=profile,
                             previous_results=tuple(profile.processing_results), max_image_pixels=settings.max_image_pixels)


def test_combined_single_image_has_one_call_typed_sources_and_candidate_materials(settings, png_bytes):
    service, _, backend, settings = setup_service(settings)
    profile = collect(service, png_bytes)
    data, result = data_of(profile)
    assert len(backend.calls) == 1 and set(backend.calls[0].context["task_ids"]) == set(IMAGE_TASK_IDS)
    assert result.status == "success" and data.checked_count == 1 and data.fusion_status == "skipped"
    image = data.image_observations[0]
    assert image.asset_id == profile.images[0].asset_id
    assert set(image.prompt_versions) == set(IMAGE_TASK_IDS) and len(image.model_metadata) == 1
    assert image.material_candidates[0].status == profile.materials[0].status == "candidate"
    assert image.visible_components[0].status == "observed" and profile.components[0].status == "detected"
    assert profile.category is None and data.object_category.status == "candidate"
    assert all(item.value is None for item in profile.measurements)
    assert profile.geometry.model_status == "not_started" and profile.geometry.mesh_path is None
    assert storage_io_path(settings.data_dir / profile.images[0].storage_path).read_bytes() == png_bytes


def test_multi_view_fusion_has_n_plus_one_calls_and_preserves_sources_conflicts(settings, png_bytes):
    service, _, backend, _ = setup_service(settings)
    profile = collect(service, png_bytes, 2)
    data, result = data_of(profile)
    assert len(backend.calls) == 3 and backend.calls[-1].images == ()
    assert data.fusion_status == "success" and result.status == "success"
    known = {image.asset_id for image in profile.images}
    assert set(data.object_category.asset_ids) == set(data.cross_view_consensus[0].asset_ids) == known
    assert data.conflicting_observations and len(data.conflicting_observations[0].claims) == 2
    assert len(data.image_observations) == 2


def test_fusion_failure_preserves_independent_observations_and_hides_raw_errors(settings, png_bytes):
    backend = TestOnlyRecognitionBackend()
    backend.fail_fusion = True
    service, _, _, _ = setup_service(settings, backend)
    profile = collect(service, png_bytes, 2)
    data, result = data_of(profile)
    assert data.fusion_status == "failed" and data.checked_count == 2 and result.status == "partial"
    assert all(item.material_candidates for item in data.image_observations)
    assert "PRIVATE" not in profile.model_dump_json()
    assert profile.materials and all(item.status == "candidate" for item in profile.materials)


@pytest.mark.parametrize("flag", ["bad_material", "foreign_source"])
def test_invalid_task_does_not_discard_other_tasks_or_accept_confirmed_claims(settings, png_bytes, flag):
    backend = TestOnlyRecognitionBackend()
    setattr(backend, flag, True)
    service, _, _, _ = setup_service(settings, backend)
    data, result = data_of(collect(service, png_bytes))
    assert result.status == "partial" and data.image_states[0].status == "partial"
    observation = data.image_observations[0]
    assert observation.task_statuses["material"] == "failed" and not observation.material_candidates
    assert observation.category_candidates and observation.visible_components and observation.viewpoint


def test_task_model_override_splits_only_the_routed_task(settings, png_bytes):
    alternate = TestOnlyRecognitionBackend()
    alternate.model_id = "alternate_test_vlm"
    service, _, default, _ = setup_service(settings, task_models={"material": alternate.model_id}, second_backend=alternate)
    data, _ = data_of(collect(service, png_bytes))
    assert len(default.calls) == len(alternate.calls) == 1
    assert default.calls[0].context["task_ids"] == [task for task in IMAGE_TASK_IDS if task != "material"]
    assert alternate.calls[0].context["task_ids"] == ["material"]
    assert {item.model_id for item in data.image_observations[0].model_metadata} == {default.model_id, alternate.model_id}


def test_explicit_independent_mode_has_five_calls_without_automatic_retries(settings, png_bytes):
    service, _, backend, _ = setup_service(settings, recognition_combine_tasks=False)
    data, _ = data_of(collect(service, png_bytes))
    assert len(backend.calls) == 5 and data.checked_count == 1
    assert [item.task_id for item in backend.calls] == list(IMAGE_TASK_IDS)


def test_unchanged_successful_images_and_fusion_are_reused_without_paid_calls(settings, png_bytes):
    service, processor, backend, settings = setup_service(settings)
    profile = collect(service, png_bytes, 2)
    assert len(backend.calls) == 3
    again = processor.process(rerun_context(profile, settings))
    data = ObjectUnderstandingData.model_validate(again.data)
    assert data.reused_count == 2 and data.fusion_status == "success"
    assert len(backend.calls) == 3


def test_partial_image_reuses_successful_tasks_and_only_retries_failed_task(settings, png_bytes):
    backend = TestOnlyRecognitionBackend()
    backend.bad_material = True
    service, processor, _, settings = setup_service(settings, backend)
    profile = collect(service, png_bytes)
    backend.bad_material = False
    result = processor.process(rerun_context(profile, settings))
    data = ObjectUnderstandingData.model_validate(result.data)
    assert len(backend.calls) == 2 and backend.calls[-1].context["task_ids"] == ["material"]
    assert data.image_states[0].status == "success" and data.image_observations[0].material_candidates


def test_changed_image_is_not_reused_even_if_archive_checksum_hint_is_stale(settings, png_bytes):
    service, processor, backend, settings = setup_service(settings)
    profile = collect(service, png_bytes)
    with Image.new("RGB", (32, 24), "red") as image:
        image.save(storage_io_path(settings.data_dir / profile.images[0].storage_path), format="PNG")
    result = processor.process(rerun_context(profile, settings))
    assert len(backend.calls) == 2 and ObjectUnderstandingData.model_validate(result.data).reused_count == 0


def test_image_failure_and_limit_do_not_affect_other_images(settings, png_bytes):
    service, processor, backend, settings = setup_service(settings, recognition_max_images=2)
    # Fail the second image after the first successful call, without knowing generated IDs.
    original = backend.analyze

    def fail_second(request):
        if len(backend.calls) == 1:
            backend.calls.append(request)
            raise RuntimeError("PRIVATE_IMAGE_DATA")
        return original(request)

    backend.analyze = fail_second
    profile = collect(service, png_bytes, 3)
    data, result = data_of(profile)
    assert [item.status for item in data.image_states] == ["success", "failed", "limit_exceeded"]
    assert data.checked_count == data.failed_count == data.skipped_count == 1 and result.status == "partial"
    assert len(profile.images) == 3 and "PRIVATE" not in profile.model_dump_json()


def test_confirmation_and_geometry_fields_are_preserved_by_adapter(settings, png_bytes):
    service, _, _, _ = setup_service(settings)
    profile = collect(service, png_bytes)
    _, result = data_of(profile)
    before = profile.model_copy(deep=True)
    before.category = "用户确认的类别"
    before.materials = [MaterialCandidate(name="陶瓷", source="user_input", status="confirmed", verification_method="用户确认")]
    before.components = [ComponentInfo(name="杯身", source="user_input", status="confirmed", verification_method="用户确认")]
    merged = ProfileAssembler().merge_result(before, result)
    assert merged.category == before.category and merged.materials == before.materials and merged.components == before.components
    assert merged.measurements == before.measurements and merged.geometry == before.geometry
    assert merged.object_id == before.object_id and merged.created_at == before.created_at


def test_valid_many_unknown_notes_do_not_discard_successful_tasks(settings, png_bytes):
    backend = TestOnlyRecognitionBackend()
    backend.many_unknowns = True
    service, _, _, _ = setup_service(settings, backend)
    data, _ = data_of(collect(service, png_bytes))
    observation = data.image_observations[0]
    assert len(observation.unknown_information) == 100
    assert sum(map(len, observation.task_unknown_information.values())) == 500


class TestMaskBackend:
    model_id = model_name = "test_mask"
    model_version = "test-only"
    capabilities = SegmentationCapabilities(box_prompt=True, quality_score=True)

    def predict(self, request):
        mask = Image.new("L", request.image.size, 0)
        ImageDraw.Draw(mask).rectangle((4, 4, 20, 18), fill=255)
        return MaskPrediction(mask, 0.9, score_type="predicted_iou")


def test_segmentation_crop_is_auxiliary_and_original_context_is_always_kept(settings, png_bytes):
    from app.models.segmentation import NormalizedBox
    segment_settings = replace(settings, segmentation_enabled=True)
    segment = ObjectSegmentationProcessor(segment_settings, backend=TestMaskBackend())
    service, _, backend, _ = setup_service(segment_settings, processors=[segment])
    profile = service.collect("物体", [ImageInput("photo.png", BytesIO(png_bytes), NormalizedBox(x0=0.1,y0=0.1,x1=0.9,y1=0.9))])
    assert profile.processing_results[0].module_name == "object_segmentation"
    assert [image.role for image in backend.calls[0].images] == ["original", "target_region"]
    assert backend.calls[0].images[0].image.size == (32, 24)
    assert backend.calls[0].images[1].image.size == (17, 15)


def test_failed_segmentation_still_allows_original_image_understanding(settings, png_bytes):
    from app.modules.base import BaseProcessor

    class FailedSegmentation(BaseProcessor):
        name = "object_segmentation"
        def process(self, context):
            return ProcessingResult(module_name=self.name, status="failed")

    service, _, backend, _ = setup_service(settings, processors=[FailedSegmentation()])
    profile = collect(service, png_bytes)
    data, result = data_of(profile)
    assert result.status == "success" and data.checked_count == 1
    assert [image.role for image in backend.calls[0].images] == ["original"]


def test_default_pipeline_http_query_download_and_missing_key(client_factory, png_bytes, monkeypatch):
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda *a, **kw: pytest.fail("No real paid calls"))
    client, settings = client_factory(recognition_enabled=True, recognition_api_key="")
    response = client.post("/api/v1/objects", data={"name": "原图仍保存"}, files=[("images", ("photo.png",png_bytes,"image/png"))])
    assert response.status_code == 201
    profile = response.json()["profile"]
    assert profile["processing_results"][-1]["module_name"] == "object_understanding"
    payload = ObjectUnderstandingData.model_validate(profile["processing_results"][-1]["data"])
    assert payload.image_states[0].status == "unavailable" and payload.checked_count == 0
    assert client.get(f"/api/v1/objects/{profile['object_id']}").json() == profile
    assert client.get(f"/api/v1/objects/{profile['object_id']}/profile.json").json() == profile
    config = client.get("/api/v1/config").json()
    assert config["recognition_enabled"] is True and "recognition_api_key" not in json.dumps(config)
    assert storage_io_path(settings.data_dir / profile["images"][0]["storage_path"]).is_file()


def test_understanding_results_roll_back_with_original_files_when_commit_fails(settings, png_bytes, monkeypatch):
    service, _, _, settings = setup_service(settings)
    monkeypatch.setattr(ObjectTransaction, "commit", lambda self: (_ for _ in ()).throw(OSError("TEST_COMMIT_FAILURE")))
    with pytest.raises(OSError):
        collect(service, png_bytes)
    for name in ("objects", ".staging"):
        directory = settings.data_dir / name
        assert not directory.exists() or not list(directory.iterdir())


def test_invalid_source_result_is_rejected_without_leaving_unvalidated_claims(settings, png_bytes):
    service, _, _, _ = setup_service(settings)
    profile = collect(service, png_bytes)
    _, result = data_of(profile)
    result.data["image_states"][0]["asset_id"] = str(uuid4())
    merged = ProfileAssembler().merge_result(profile, result)
    assert merged.processing_results[-1].status == "failed" and merged.processing_results[-1].data == {}
    assert merged.materials == profile.materials and merged.components == profile.components


def test_provider_long_warning_does_not_discard_valid_observation(settings, png_bytes):
    backend = TestOnlyRecognitionBackend()
    analyze = backend.analyze
    def warning_result(request):
        result = analyze(request)
        return RecognitionResult(result.data, result.model_id, result.model_version, result.processing_time,
                                 warnings=("a" * 2000,), metadata=result.metadata)
    backend.analyze = warning_result
    service, _, _, _ = setup_service(settings, backend)
    data, _ = data_of(collect(service, png_bytes))
    assert data.checked_count == 1 and len(data.image_observations[0].warnings[0]) <= 500


def test_fusion_warning_aggregation_is_bounded_without_discarding_views(settings, png_bytes):
    backend = TestOnlyRecognitionBackend()
    analyze = backend.analyze
    def warning_result(request):
        result = analyze(request)
        if request.task_id == "multi_view_fusion":
            result.data["warnings"] = [f"fusion warning {i}" for i in range(100)]
        return result
    backend.analyze = warning_result
    service, _, _, _ = setup_service(settings, backend, recognition_max_images=2)
    data, _ = data_of(collect(service, png_bytes, 3))
    assert data.checked_count == 2 and data.fusion_status == "success" and len(data.warnings) == 100


def test_large_analysis_copy_is_bounded_without_modifying_original(settings):
    with Image.new("RGB", (900, 600), "white") as image, BytesIO() as buf:
        image.save(buf, format="PNG")
        original = buf.getvalue()
    service, _, backend, settings = setup_service(settings, recognition_max_image_side=256)
    profile = collect(service, original)
    data, _ = data_of(profile)
    assert max(backend.calls[0].images[0].image.size) <= 256 and data.checked_count == 1
    assert storage_io_path(settings.data_dir / profile.images[0].storage_path).read_bytes() == original


def test_all_images_failed_still_saves_originals_and_safe_failure_states(settings, png_bytes):
    backend = TestOnlyRecognitionBackend()
    def broken(request):
        backend.calls.append(request)
        raise RuntimeError("PRIVATE_KEY PRIVATE_RESPONSE")
    backend.analyze = broken
    service, _, _, settings = setup_service(settings, backend)
    profile = collect(service, png_bytes, 2)
    data, result = data_of(profile)
    assert data.checked_count == 0 and data.failed_count == 2 and result.status == "failed"
    assert len(backend.calls) == 2 and "PRIVATE" not in profile.model_dump_json()
    assert all(storage_io_path(settings.data_dir / asset.storage_path).is_file() for asset in profile.images)


def test_disabled_direct_processor_does_not_resolve_models_or_images(settings):
    processor = ObjectUnderstandingProcessor(settings)
    result = processor.process(ProcessingContext(images=(), data_dir=settings.data_dir))
    assert result.status == "skipped" and ObjectUnderstandingData.model_validate(result.data).input_count == 0


def test_model_specific_prompt_change_only_reexecutes_its_own_task(settings, png_bytes, monkeypatch):
    from app.prompts import object_understanding as prompts
    monkeypatch.setattr(prompts, "_MODEL_PROMPTS", {})
    service, processor, backend, settings = setup_service(settings)
    profile = collect(service, png_bytes)
    original_other = prompts.compose_prompts(("appearance",), model_id=backend.model_id)
    base = prompts.get_prompt("material")
    prompts.register_prompt_adapter(backend.model_id, "material", version="test-adapted-2", text=base.text + "\n模型特定措辞测试。")
    result = processor.process(rerun_context(profile, settings))
    data = ObjectUnderstandingData.model_validate(result.data)
    assert len(backend.calls) == 2 and backend.calls[-1].context["task_ids"] == ["material"]
    assert data.image_observations[0].prompt_versions["material"] == "test-adapted-2"
    assert prompts.get_prompt("material").version == base.version
    assert prompts.compose_prompts(("appearance",), model_id=backend.model_id) == original_other


def test_crop_input_policy_change_invalidates_original_only_cached_result(settings, png_bytes):
    from app.models.segmentation import NormalizedBox
    segment_settings = replace(settings, segmentation_enabled=True)
    segment = ObjectSegmentationProcessor(segment_settings, backend=TestMaskBackend())
    service, processor, backend, settings = setup_service(segment_settings, processors=[segment], recognition_max_images=1)
    profile = service.collect("物体", [ImageInput("photo.png", BytesIO(png_bytes), NormalizedBox(x0=0.1,y0=0.1,x1=0.9,y1=0.9))])
    assert len(backend.calls[0].images) == 1 and not backend.calls[0].context["target_region_is_auxiliary"]
    changed = replace(settings, recognition_max_images=2)
    processor = ObjectUnderstandingProcessor(changed, router=RecognitionRouter(changed, processor.router.registry))
    result = processor.process(rerun_context(profile, changed))
    assert ObjectUnderstandingData.model_validate(result.data).reused_count == 0
    assert len(backend.calls) == 2 and len(backend.calls[-1].images) == 2
    assert backend.calls[-1].context["target_region_is_auxiliary"]
