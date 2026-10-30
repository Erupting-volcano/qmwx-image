"""Tasks and prompt isolation use a test backend, never a paid model service."""

from io import BytesIO
from uuid import uuid4

from PIL import Image
import pytest

from app.models.object_understanding import (
    CategoryCandidate, ComponentObservation, ImageObservation, MaterialObservation,
)
from app.modules.object_understanding.tasks import TASKS, compose_output_schema, get_task
from app.modules.recognition_backend import (
    RecognitionCapabilities, RecognitionImage, RecognitionResult, UnsupportedRecognitionCapability,
)
from app.prompts.object_understanding import compose_prompts, get_prompt


def sourced(asset, **fields):
    return {"asset_ids": [str(asset)], "evidence": "测试可见证据", **fields}


def task_outputs(asset):
    return {
        "classification": {"category_candidates": [sourced(asset, category="马克杯", broad_category="饮用容器")],
                           "object_name": sourced(asset, text="杯子", status="candidate"),
                           "possible_uses": [sourced(asset, text="喝水", status="candidate")]},
        "appearance": {"appearance": {"colors": [sourced(asset, text="白色", role="primary")],
                                       "shape_description": sourced(asset, text="杯身近圆柱状")}},
        "material": {"material_candidates": [sourced(asset, material_name="陶瓷", uncertainty="也可能是白色涂层材料")]},
        "components": {"visible_components": [sourced(asset, component_name="把手", description="可见环形握持部分")]},
        "viewpoint": {"viewpoint": sourced(asset, viewpoint="perspective")},
    }


class TestBackend:
    __test__ = False
    model_id = "task_test_backend"
    model_version = "test-only"
    capabilities = RecognitionCapabilities(image_input=True, structured_output=True)

    def __init__(self, output):
        self.output, self.calls = output, []

    def analyze(self, request):
        self.calls.append(request)
        return RecognitionResult(data=self.output, model_id=self.model_id, model_version=self.model_version,
                                 processing_time=0.01, metadata={"task_test_backend": {"test_fixture": True}})


@pytest.mark.parametrize("task_id", ["classification", "appearance", "material", "components", "viewpoint"])
def test_independent_tasks_execute_with_uniform_backend_and_typed_output(task_id):
    asset = uuid4()
    backend = TestBackend(task_outputs(asset)[task_id])
    with Image.new("RGB", (32, 24)) as image:
        execution = get_task(task_id).execute(backend, {"asset_id": asset, "object_name_hint": "用户提示并非事实"},
                                               (RecognitionImage(asset, image),))
    assert len(backend.calls) == 1 and backend.calls[0].task_id == task_id
    assert "JSON" in backend.calls[0].instruction and backend.calls[0].output_schema
    assert execution.model_metadata.model_id == "task_test_backend"
    assert execution.model_metadata.task_ids == [task_id]
    assert execution.prompt_versions == {task_id: get_prompt(task_id).version}
    assert execution.data.model_dump()[next(iter(task_outputs(asset)[task_id]))]


def test_task_output_rejects_foreign_sources_before_archiving():
    asset, foreign = uuid4(), uuid4()
    with pytest.raises(ValueError):
        get_task("material").validate_output(task_outputs(foreign)["material"], {asset})


def test_invalid_provider_payload_is_not_embedded_in_validation_errors():
    asset = uuid4()
    with pytest.raises(ValueError) as caught:
        get_task("material").validate_output({"material_candidates": "PRIVATE_PROVIDER_RESPONSE_BASE64"}, {asset})
    assert "PRIVATE_PROVIDER_RESPONSE_BASE64" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_output_tasks_do_not_share_schemas_or_mutable_state():
    asset = uuid4()
    outputs = task_outputs(asset)
    validated = {task_id: get_task(task_id).validate_output(data, {asset}) for task_id, data in outputs.items()}
    outputs["material"]["material_candidates"][0]["status"] = "confirmed"
    with pytest.raises(ValueError):
        get_task("material").validate_output(outputs["material"], {asset})
    assert validated["appearance"].appearance.colors[0].text == "白色"
    assert validated["material"].material_candidates[0].status == "candidate"
    assert len({task.output_model for task in TASKS.values()}) == 6


def test_prompts_are_independent_versioned_and_dynamically_composed(monkeypatch):
    from app.prompts.object_understanding import classification

    before_appearance = compose_prompts(("appearance",))
    before_combined = compose_prompts(("classification", "appearance"))
    monkeypatch.setattr(classification, "VERSION", "test-version")
    monkeypatch.setattr(classification, "TEXT", classification.TEXT + "\n仅测试新的分类说明。")
    assert compose_prompts(("appearance",)) == before_appearance
    after = compose_prompts(("classification", "appearance"))
    assert after.fingerprint != before_combined.fingerprint
    assert after.versions == {"classification": "test-version", "appearance": before_appearance.versions["appearance"]}
    assert "仅测试新的分类说明" in after.instruction


def test_compound_schema_references_resolve_at_root_without_task_name_collisions():
    schema = compose_output_schema(("classification", "appearance", "material", "components", "viewpoint"))
    assert set(schema["properties"]) == set(schema["required"])

    def check(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].removeprefix("#/$defs/") in schema["$defs"]
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)
    assert "classification_CategoryCandidate" in schema["$defs"]
    assert "material_MaterialObservation" in schema["$defs"]


def test_missing_image_and_unsupported_backend_are_rejected_before_call():
    asset = uuid4()
    backend = TestBackend({})
    with pytest.raises(ValueError):
        get_task("classification").execute(backend, {"asset_id": asset})
    backend.capabilities = RecognitionCapabilities(structured_output=True)
    with Image.new("RGB", (10, 10)) as image, pytest.raises(UnsupportedRecognitionCapability):
        get_task("classification").execute(backend, {"asset_id": asset}, (RecognitionImage(asset, image),))
    assert backend.calls == []


def test_crop_alone_cannot_replace_original_image_context():
    asset = uuid4()
    backend = TestBackend(task_outputs(asset)["appearance"])
    with Image.new("RGB", (10, 10)) as image, pytest.raises(ValueError, match="原图"):
        get_task("appearance").execute(backend, {"asset_id": asset},
                                       (RecognitionImage(asset, image, role="target_region"),))
    assert backend.calls == []


def test_multi_view_fusion_is_a_separate_text_context_call_and_preserves_conflicts():
    first, second = uuid4(), uuid4()
    views = [ImageObservation(
        asset_id=asset,
        category_candidates=[CategoryCandidate(**sourced(asset, category="杯子"))],
        material_candidates=[MaterialObservation(**sourced(asset, material_name="陶瓷"))],
        visible_components=[ComponentObservation(**sourced(asset, component_name="把手", description="环形部分"))],
    ) for asset in (first, second)]
    backend = TestBackend({
        "object_category": {"category": "杯子", "evidence": "两图一致", "asset_ids": [str(first), str(second)]},
        "cross_view_consensus": [{"text": "可见把手", "evidence": "共同支持", "asset_ids": [str(first), str(second)]}],
        "conflicting_observations": [{"field": "颜色", "description": "保留色差", "claims": [
            sourced(first, text="白色"), sourced(second, text="淡黄色")]}],
        "unknown_information": ["真实材料和尺寸未确认"],
    })
    result = get_task("multi_view_fusion").execute(backend, {"image_observations": views})
    assert len(backend.calls) == 1 and backend.calls[0].images == ()
    assert backend.calls[0].context["image_observations"][0]["asset_id"] == str(first)
    assert result.data.object_category.status == "candidate"
    assert len(result.data.conflicting_observations[0].claims) == 2
    assert result.data.unknown_information == ["真实材料和尺寸未确认"]


@pytest.mark.parametrize("task_ids", [(), ("material", "material"), ("not_a_task",)])
def test_unknown_or_invalid_task_composition_is_explicit(task_ids):
    with pytest.raises(ValueError):
        compose_prompts(task_ids)
    with pytest.raises(ValueError):
        compose_output_schema(task_ids)
