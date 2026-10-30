"""Framework/storage tests only; synthetic backend outputs are not real inference."""

import base64
from dataclasses import replace
from io import BytesIO
import json
import traceback

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from app.config import Settings
from app.main import create_app
from app.models.segmentation import SegmentationData, SegmentationImage
from app.modules import segmentation_factory as factory
from app.modules.baidu_backend import BaiduSegmentationBackend, TOKEN_URL
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.modules.segmentation_backend import MaskCandidate, MaskPrediction, SegmentationCapabilities
from app.services.image_collection import ImageCollectionService, ImageInput
from app.storage import local_storage
from app.storage.local_storage import ObjectTransaction, resolve_mask_path, storage_io_path


def record(profile):
    result = next(item for item in profile.processing_results if item.module_name == "object_segmentation")
    return SegmentationData.model_validate(result.data).images[0]


def post(client, content):
    return client.post("/api/v1/objects", data={"name": "框架回归", "segmentation_boxes": "[[0.1,0.1,0.9,0.9]]"},
                       files=[("images", ("photo.png", content, "image/png"))])


class FrameworkTestBackend:
    """Explicitly test-only: exercise protocol and artifacts, not model accuracy."""

    model_id = "framework_test"
    model_name = "framework_test"
    model_version = "test-only"
    capabilities = SegmentationCapabilities(box_prompt=True, multiple_masks=True, quality_score=True)

    def __init__(self, *, score=0.9, score_type="predicted_iou", candidates=False):
        self.score, self.score_type, self.candidates = score, score_type, candidates

    def predict(self, request):
        mask = Image.new("L", request.image.size, 0)
        ImageDraw.Draw(mask).rectangle((4, 4, 15, 15), fill=255)
        candidates = ()
        if self.candidates:
            other = Image.new("L", request.image.size, 0)
            ImageDraw.Draw(other).rectangle((6, 5, 20, 17), fill=255)
            candidates = (MaskCandidate(other, 2.3, "matching_logit", ("仅测试候选存储",)),)
        return MaskPrediction(
            mask, self.score, score_type=self.score_type, candidates=candidates,
            model_id=self.model_id, model_version=self.model_version, processing_time=0.125,
            extensions={self.model_id: {"mode": "test-fixture", "small_state": {"iteration": 1}}},
        )


def register_fixture(monkeypatch, backend=None, builder=None):
    registry = factory.SegmentationRegistry()
    backend = backend or FrameworkTestBackend()
    registry.register(factory.BackendRegistration(
        model_id=backend.model_id, model_name=backend.model_name, model_version=backend.model_version,
        name="Test fixture only", description="No real model inference", deployment="local", status="available",
        capabilities=backend.capabilities, builder=builder or (lambda settings: backend),
    ))
    monkeypatch.setattr(factory, "SEGMENTATION_REGISTRY", registry)
    return backend


@pytest.mark.parametrize("backend_id,phrase", [("sam2.1_hiera_tiny", "尚未"), ("unregistered_model", "未注册")])
def test_selected_unimplemented_or_unknown_model_preserves_basic_functions(client_factory, png_bytes, monkeypatch,
                                                                         backend_id, phrase):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network or fallback is allowed")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)
    client, settings = client_factory(segmentation_enabled=True, segmentation_backend=backend_id)
    config = client.get("/api/v1/config").json()
    assert config["segmentation_backend"] == backend_id
    assert config["segmentation_status"] == ("not_implemented" if backend_id.startswith("sam") else "unavailable")
    assert phrase in config["segmentation_message"]
    response = post(client, png_bytes)
    assert response.status_code == 201, response.text
    from app.models.object_profile import ObjectProfile
    profile = ObjectProfile.model_validate(response.json()["profile"])
    item = record(profile)
    assert item.model_id == item.model_name == backend_id
    assert item.status == "unavailable" and item.error_id and item.mask_path is None
    assert phrase in item.warnings[0]
    assert profile.processing_results[0].status == profile.processing_results[1].status == "success"
    assert storage_io_path(settings.data_dir / profile.images[0].storage_path).read_bytes() == png_bytes
    assert client.get(f"/api/v1/objects/{profile.object_id}/profile.json").json() == profile.model_dump(mode="json")
    without_box = client.post("/api/v1/objects", data={"name": "未提供框"},
                              files=[("images", ("photo.png", png_bytes, "image/png"))]).json()["profile"]
    assert without_box["processing_results"][-1]["data"]["images"][0]["status"] == "unavailable"
    assert not list(settings.data_dir.rglob("masks/*.png"))


def test_disabled_model_never_initializes_even_with_missing_configuration(client_factory, png_bytes, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled model must never initialize")

    monkeypatch.setattr("app.modules.object_segmentation.create_backend", forbidden)
    client, _ = client_factory(segmentation_backend="sam2.1_hiera_tiny", segmentation_enabled=False,
                              sam2_checkpoint_path="missing-model.pt")
    assert client.get("/api/v1/config").json()["segmentation_status"] == "disabled"
    item = post(client, png_bytes).json()["profile"]["processing_results"][-1]["data"]["images"][0]
    assert item["status"] == "disabled" and item["model_id"] == "sam2.1_hiera_tiny"


def test_default_baidu_adapter_runs_through_factory_and_persists_typed_output(client_factory, monkeypatch):
    calls = []

    def mock_http(self, request, max_bytes):
        calls.append(request)
        if request.full_url == TOKEN_URL:
            return {"access_token": "TEST_ONLY_TOKEN", "expires_in": 3600}
        body = json.loads(request.data)
        assert body["method"] == "control"
        assert body["position"] == [[[25, 25], [231, 231]]]
        with Image.new("L", (256, 256), 0) as mask, BytesIO() as buf:
            ImageDraw.Draw(mask).rectangle((40, 40, 180, 200), fill=255)
            mask.save(buf, format="PNG")
            return {"image": base64.b64encode(buf.getvalue()).decode()}

    monkeypatch.setattr(BaiduSegmentationBackend, "_request_json", mock_http)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda *a, **kw: pytest.fail("No live HTTP"))
    client, _ = client_factory(segmentation_enabled=True, segmentation_backend="baidu",
                               segmentation_api_key="TEST_KEY", segmentation_secret_key="TEST_SECRET")
    with Image.new("RGB", (256, 256), "white") as image, BytesIO() as buf:
        image.save(buf, format="PNG")
        content = buf.getvalue()
    response = post(client, content)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    item = SegmentationData.model_validate(profile["processing_results"][-1]["data"]).images[0]
    assert len(calls) == 2 and item.model_id == "baidu"
    assert item.model_name == "baidu_intelligent_segmentation" and item.model_version == "service-unversioned"
    assert item.status == "needs_confirmation" and item.score is item.score_type is item.confidence is None
    assert item.extensions == {"baidu": {"api_mode": "control"}} and item.processing_time >= 0
    base = f"/api/v1/objects/{profile['object_id']}"
    assert client.get(base).json() == client.get(base + "/profile.json").json() == profile
    assert client.get(base + f"/images/{item.asset_id}/mask").status_code == 200
    assert "TEST_SECRET" not in json.dumps(profile) and "TEST_KEY" not in client.get("/api/v1/config").text


@pytest.mark.parametrize("score,kind,status", [(0.9, "predicted_iou", "success"),
                                               (0.9, "text_match", "needs_confirmation"),
                                               (2.3, "matching_logit", "needs_confirmation")])
def test_model_scores_keep_definitions_and_are_not_compared(settings, png_bytes, monkeypatch, score, kind, status):
    register_fixture(monkeypatch, FrameworkTestBackend(score=score, score_type=kind))
    selected = replace(settings, segmentation_enabled=True, segmentation_backend="framework_test")
    service = ImageCollectionService(selected)
    from app.models.segmentation import NormalizedBox
    profile = service.collect("测试分数", [ImageInput("photo.png", BytesIO(png_bytes),
                                                 NormalizedBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9))])
    item = record(profile)
    assert (item.model_id, item.model_version, item.score, item.score_type, item.status) == (
        "framework_test", "test-only", score, kind, status)
    assert item.confidence == (score if kind == "predicted_iou" and 0 <= score <= 1 else None)
    assert item.extensions["framework_test"]["small_state"] == {"iteration": 1}


def test_candidates_follow_existing_transaction_and_ownership_checks(settings, png_bytes, monkeypatch):
    register_fixture(monkeypatch, FrameworkTestBackend(candidates=True))
    selected = replace(settings, segmentation_enabled=True, segmentation_backend="framework_test")
    with TestClient(create_app(selected)) as client:
        response = post(client, png_bytes)
        assert response.status_code == 201, response.text
        profile = response.json()["profile"]
        item = SegmentationData.model_validate(profile["processing_results"][-1]["data"]).images[0]
        assert len(item.candidates) == 1 and item.status == "needs_confirmation"
        assert item.candidates[0].score == 2.3 and item.candidates[0].score_type == "matching_logit"
        paths = []
        for mask in item.stored_masks():
            paths.append(resolve_mask_path(selected.data_dir, profile["object_id"], item.asset_id, mask.mask_path,
                                           (item.width, item.height), expected_bbox=mask.bbox,
                                           expected_area_ratio=mask.area_ratio))
        assert len(paths) == len(set(paths)) == 2 and all(path.is_file() for path in paths)
        assert list((selected.data_dir / ".staging").iterdir()) == []
        assert client.get(f"/api/v1/objects/{profile['object_id']}/profile.json").json() == profile


def test_failed_primary_write_cleans_already_staged_candidates(settings, png_bytes, monkeypatch):
    register_fixture(monkeypatch, FrameworkTestBackend(candidates=True))
    selected = replace(settings, segmentation_enabled=True, segmentation_backend="framework_test")
    original_save = ObjectTransaction.save_mask
    calls = []

    def save_then_fail(self, asset_id, mask):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("TEST_ONLY_STORAGE_FAILURE")
        return original_save(self, asset_id, mask)

    monkeypatch.setattr(ObjectTransaction, "save_mask", save_then_fail)
    with TestClient(create_app(selected)) as client:
        response = post(client, png_bytes)
        assert response.status_code == 201
        profile = response.json()["profile"]
        item = SegmentationData.model_validate(profile["processing_results"][-1]["data"]).images[0]
        assert item.status == "failed" and item.candidates == [] and item.mask_path is None
        assert not list(selected.data_dir.rglob("masks/*.png"))
        assert list((selected.data_dir / ".staging").iterdir()) == []
        assert storage_io_path(selected.data_dir / profile["images"][0]["storage_path"]).read_bytes() == png_bytes


def test_candidate_transaction_commit_failure_rolls_back_every_file(settings, png_bytes, monkeypatch):
    register_fixture(monkeypatch, FrameworkTestBackend(candidates=True))
    selected = replace(settings, segmentation_enabled=True, segmentation_backend="framework_test")
    original_replace = local_storage._atomic_replace

    def fail_commit(source, destination):
        if source.is_dir():
            raise OSError("TEST_ONLY_COMMIT_FAILURE")
        return original_replace(source, destination)

    monkeypatch.setattr(local_storage, "_atomic_replace", fail_commit)
    with TestClient(create_app(selected), raise_server_exceptions=False) as client:
        assert post(client, png_bytes).status_code == 500
    for name in ("objects", ".staging"):
        directory = selected.data_dir / name
        assert not directory.exists() or not list(directory.iterdir())


def test_failed_initialization_is_cached_without_retaining_tracebacks(settings, monkeypatch):
    attempts = []

    def broken(settings):
        attempts.append(True)
        raise RuntimeError("PRIVATE_CONFIGURATION_VALUE")

    register_fixture(monkeypatch, builder=broken)
    processor = ObjectSegmentationProcessor(replace(settings, segmentation_backend="framework_test"))
    exceptions = []
    for _ in range(3):
        try:
            processor.backend
        except Exception as exc:
            exceptions.append(exc)
    assert len(attempts) == 1 and len(set(map(id, exceptions))) == 3
    assert len(set(len(traceback.extract_tb(exc.__traceback__)) for exc in exceptions)) == 1
    assert all("PRIVATE_CONFIGURATION_VALUE" not in "".join(traceback.format_exception(exc)) for exc in exceptions)


def test_historical_records_and_new_model_names_have_safe_defaults():
    old = SegmentationImage(asset_id="72f04a91-2645-4bd8-944e-208d8e24da77", filename="old.png", status="disabled")
    payload = old.model_dump(exclude={"model_id", "model_version", "score", "score_type", "processing_time",
                                      "extensions", "candidates"})
    restored = SegmentationImage.model_validate(payload)
    assert restored.model_name == "baidu_intelligent_segmentation"
    assert restored.model_id is restored.model_version is restored.score_type is restored.processing_time is None
    assert restored.extensions == {} and restored.candidates == []
    assert SegmentationImage.model_validate({**payload, "model_name": "future_model_v3"}).model_name == "future_model_v3"


@pytest.mark.parametrize("extensions", [{"sam2": {"data": b"binary"}}, {"bad namespace": {}},
                                         {"baidu": {"large": "x" * 17000}}, {"baidu": {"nan": float("nan")}}])
def test_model_extensions_reject_binary_unbounded_or_invalid_json(extensions):
    with pytest.raises(ValueError):
        SegmentationImage(asset_id="72f04a91-2645-4bd8-944e-208d8e24da77", filename="old.png",
                          status="disabled", extensions=extensions)


def test_undeclared_candidate_score_is_rejected_before_any_mask_is_saved(settings, png_bytes, monkeypatch):
    backend = FrameworkTestBackend(score=None, score_type=None, candidates=True)
    backend.capabilities = SegmentationCapabilities(box_prompt=True, multiple_masks=True)
    register_fixture(monkeypatch, backend)
    selected = replace(settings, segmentation_enabled=True, segmentation_backend="framework_test")
    with TestClient(create_app(selected)) as client:
        response = post(client, png_bytes)
    item = response.json()["profile"]["processing_results"][-1]["data"]["images"][0]
    assert item["status"] == "failed" and item["mask_path"] is None
    assert not list(selected.data_dir.rglob("masks/*.png"))


def test_box_upload_refuses_a_registered_point_only_model(settings, png_bytes, monkeypatch):
    backend = FrameworkTestBackend()
    backend.capabilities = SegmentationCapabilities(point_prompt=True)

    def forbidden(request):
        pytest.fail("Unsupported prompt must be rejected before prediction")

    backend.predict = forbidden
    register_fixture(monkeypatch, backend)
    selected = replace(settings, segmentation_enabled=True, segmentation_backend="framework_test")
    with TestClient(create_app(selected)) as client:
        response = post(client, png_bytes)
    item = response.json()["profile"]["processing_results"][-1]["data"]["images"][0]
    assert item["status"] == "unavailable" and "不支持" in item["warnings"][0]
    assert not list(selected.data_dir.rglob("masks/*.png"))
