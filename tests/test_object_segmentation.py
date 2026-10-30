"""Mask contracts use injected predictions; paid cloud inference is opt-in."""

from dataclasses import replace
from io import BytesIO
import os
from uuid import uuid4

from PIL import Image, ImageDraw
from pydantic import ValidationError
import pytest

from app.config import Settings
from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQuality, ImageQualityData
from app.models.processing_result import ProcessingResult
from app.models.segmentation import NormalizedBox, SegmentationData, SegmentationImage
from app.modules import image_metadata
from app.modules.base import ProcessingContext
from app.modules.baidu_backend import BaiduSegmentationBackend
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.modules.segmentation_backend import (
    MaskCandidate, MaskPrediction, SegmentationAPIError, SegmentationCapabilities, SegmentationUnavailable,
)
from app.storage.local_storage import storage_io_path


BOX = NormalizedBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75)


def image_bytes(size=(32, 24), *, orientation=None):
    buffer = BytesIO()
    image = Image.new("RGB", size, (120, 130, 140))
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


class RecordingStore:
    def __init__(self, root, object_id):
        self.root, self.object_id, self.paths = root, object_id, []

    def save_mask(self, asset_id, mask):
        path = f"objects/{self.object_id}/masks/{asset_id}_{uuid4()}.png"
        destination = self.root / path
        storage_io_path(destination.parent).mkdir(parents=True, exist_ok=True)
        mask.save(storage_io_path(destination), format="PNG")
        self.paths.append(path)
        return path


class PredictingBackend:
    model_id = "test_only"
    model_name = "baidu_intelligent_segmentation"
    model_version = "test-fixture"
    capabilities = SegmentationCapabilities(box_prompt=True, multiple_masks=True, quality_score=True)

    def __init__(self, outcome="normal", score=0.95):
        self.outcome, self.score, self.calls = outcome, score, []

    def predict(self, request):
        image, box = request.image, request.prompt.box
        self.calls.append((image.size, box))
        if self.outcome == "error" or (self.outcome == "second_error" and len(self.calls) == 2):
            raise RuntimeError("PRIVATE_MODEL_PATH_C:\\private\\weights")
        if self.outcome == "unavailable":
            raise SegmentationUnavailable("云端分割服务不可用，请检查服务开通状态。")
        mask = Image.new("L", image.size, 255 if self.outcome == "full" else 0)
        if self.outcome not in {"empty", "full"}:
            x0, y0, x1, y1 = box
            ImageDraw.Draw(mask).rectangle((int(x0), int(y0), int(x1) - 1, int(y1) - 1), fill=255)
        if self.outcome == "gray":
            mask.putpixel((0, 0), 128)
        if self.outcome == "wrong_size":
            mask = mask.resize((3, 3))
        return MaskPrediction(mask, self.score, score_type="predicted_iou")


def make_context(tmp_path, contents=None, *, boxes=True, max_pixels=40_000_000):
    object_id = uuid4()
    contents = contents or [image_bytes()]
    assets = []
    for index, content in enumerate(contents):
        asset_id = uuid4()
        relative = f"objects/{object_id}/images/{asset_id}.png"
        path = tmp_path / relative
        storage_io_path(path.parent).mkdir(parents=True, exist_ok=True)
        storage_io_path(path).write_bytes(content)
        # Deliberately inaccurate hints cannot stand in for actual image checks.
        assets.append(ImageAsset(
            asset_id=asset_id, filename="same.png", storage_path=relative, width=1, height=1,
            image_format="PNG", mime_type="image/png", size_bytes=1,
        ))
    store = RecordingStore(tmp_path, object_id)
    return ProcessingContext(
        images=tuple(assets), data_dir=tmp_path, max_image_pixels=max_pixels,
        artifact_store=store, segmentation_boxes={asset.asset_id: BOX for asset in assets} if boxes else {},
    )


def processor(tmp_path, backend):
    return ObjectSegmentationProcessor(Settings(data_dir=tmp_path, segmentation_enabled=True), backend=backend)


def parsed(result):
    return SegmentationData.model_validate(result.data)


def test_normal_prediction_records_real_mask_bbox_area_and_asset_id(tmp_path):
    context = make_context(tmp_path)
    original_path = context.resolve_asset(context.images[0])
    original_content = original_path.read_bytes()
    backend = PredictingBackend()
    result = processor(tmp_path, backend).process(context)
    data = parsed(result)
    item = data.images[0]
    assert result.status == "success"
    assert (data.input_count, data.success_count, data.review_count, data.failed_count, data.skipped_count) == (1, 1, 0, 0, 0)
    assert item.asset_id == context.images[0].asset_id
    assert item.status == "success"
    assert item.bbox == (8, 6, 24, 18)
    assert item.area_ratio == pytest.approx(0.25)
    assert item.confidence == pytest.approx(0.95)
    assert backend.calls == [((32, 24), (8.0, 6.0, 24.0, 18.0))]
    assert "人工核对" in item.warnings[0] and "目标" in item.warnings[0]
    assert "mask" not in result.data  # No pixel arrays are embedded in JSON.
    with Image.open(storage_io_path(tmp_path / item.mask_path)) as saved:
        assert saved.mode == "L" and saved.size == (32, 24)
        assert {index for index, count in enumerate(saved.histogram()) if count} == {0, 255}
        assert saved.getbbox() == item.bbox
    assert original_path.read_bytes() == original_content


def test_empty_prediction_is_not_found_without_a_fake_mask(tmp_path):
    context = make_context(tmp_path)
    result = processor(tmp_path, PredictingBackend("empty")).process(context)
    data = parsed(result)
    assert result.status == "failed"
    assert data.failed_count == 1
    assert data.images[0].status == "not_found"
    assert data.images[0].mask_path is None and data.images[0].area_ratio == 0
    assert data.images[0].error_id
    assert context.artifact_store.paths == []


def test_empty_prediction_retains_provider_coordinate_mapping_warning(tmp_path):
    mapping_warning = "云端掩码已缩放回原图坐标，请复查边界。"

    class EmptyCloudBackend(PredictingBackend):
        def predict(self, request):
            prediction = super().predict(request)
            return MaskPrediction(prediction.mask, None, warnings=(mapping_warning,))

    context = make_context(tmp_path)
    result = processor(tmp_path, EmptyCloudBackend("empty")).process(context)
    data = parsed(result)
    item = data.images[0]
    assert result.status == "failed" and data.failed_count == 1
    assert item.status == "not_found" and item.area_ratio == 0
    assert item.mask_path is None and item.error_id == result.errors[0].error_id
    assert "没有" in item.warnings[0]
    assert mapping_warning in item.warnings
    assert any(mapping_warning in warning for warning in result.warnings)
    assert context.artifact_store.paths == []


@pytest.mark.parametrize("outcome,score", [("full", 0.95), ("normal", 0.2), ("normal", None), ("normal", float("nan"))])
def test_full_masks_or_unreliable_quality_scores_require_confirmation(tmp_path, outcome, score):
    result = processor(tmp_path, PredictingBackend(outcome, score)).process(make_context(tmp_path))
    data = parsed(result)
    assert result.status == "partial"
    assert data.review_count == 1 and data.success_count == 0
    assert data.images[0].status == "needs_confirmation"
    assert data.images[0].mask_path
    assert data.images[0].warnings
    assert data.images[0].error_id is None


@pytest.mark.parametrize("outcome", ["error", "gray", "wrong_size"])
def test_invalid_backend_output_is_isolated_with_safe_errors(tmp_path, outcome):
    context = make_context(tmp_path)
    result = processor(tmp_path, PredictingBackend(outcome)).process(context)
    data = parsed(result)
    assert result.status == "failed" and data.failed_count == 1
    assert data.images[0].status == "failed"
    assert data.images[0].error_id == result.errors[0].error_id
    assert "PRIVATE_MODEL_PATH" not in result.model_dump_json()
    assert context.artifact_store.paths == []


def test_safe_api_failure_explanation_is_visible_in_warnings_and_errors(tmp_path):
    safe_message = "百度分割API请求过于频繁，请检查额度后稍后重试；本次未自动重试。"

    class RateLimitedBackend(PredictingBackend):
        def predict(self, request):
            raise SegmentationAPIError(safe_message)

    context = make_context(tmp_path)
    result = processor(tmp_path, RateLimitedBackend()).process(context)
    data = parsed(result)
    item = data.images[0]
    assert result.status == "failed" and data.failed_count == 1
    assert item.status == "failed" and item.mask_path is None
    assert item.warnings == [safe_message]
    assert any(safe_message in warning for warning in result.warnings)
    assert result.errors[0].message == safe_message
    assert result.errors[0].code == "segmentation_failed"
    assert result.errors[0].error_id == item.error_id
    assert result.errors[0].asset_id == context.images[0].asset_id
    assert context.artifact_store.paths == []


def test_second_image_failure_keeps_first_mask_and_distinct_resource_ids(tmp_path):
    context = make_context(tmp_path, [image_bytes(), image_bytes()])
    result = processor(tmp_path, PredictingBackend("second_error")).process(context)
    data = parsed(result)
    assert result.status == "partial"
    assert (data.success_count, data.failed_count) == (1, 1)
    assert [item.asset_id for item in data.images] == [item.asset_id for item in context.images]
    assert [item.status for item in data.images] == ["success", "failed"]
    assert len(context.artifact_store.paths) == 1


def test_disabled_or_missing_box_does_not_call_the_model(tmp_path):
    backend = PredictingBackend("error")
    disabled = ObjectSegmentationProcessor(Settings(data_dir=tmp_path, segmentation_enabled=False), backend=backend)
    result = disabled.process(make_context(tmp_path))
    assert result.status == "skipped"
    assert parsed(result).images[0].status == "disabled"
    result = processor(tmp_path, backend).process(make_context(tmp_path, boxes=False))
    assert result.status == "needs_input"
    assert parsed(result).images[0].status == "needs_input"
    assert backend.calls == []


def test_missing_credentials_and_api_unavailability_are_explicit(tmp_path):
    context = make_context(tmp_path)
    missing = ObjectSegmentationProcessor(Settings(
        data_dir=tmp_path, segmentation_enabled=True, segmentation_api_key="", segmentation_secret_key="",
    ))
    result = missing.process(context)
    assert result.status == "skipped"
    assert parsed(result).images[0].status == "unavailable"
    assert "API Key" in result.errors[0].message and "Secret Key" in result.errors[0].message
    result = processor(tmp_path, PredictingBackend("unavailable")).process(context)
    assert parsed(result).skipped_count == 1
    assert parsed(result).images[0].mask_path is None
    assert result.errors[0].error_id


@pytest.mark.parametrize("content", [b"not an image", image_bytes()[:35]])
def test_corrupt_image_never_reaches_model(tmp_path, content):
    backend = PredictingBackend()
    result = processor(tmp_path, backend).process(make_context(tmp_path, [content]))
    assert parsed(result).images[0].status == "failed"
    assert backend.calls == []


def test_invalid_path_and_actual_pixel_limit_are_checked_before_model(tmp_path):
    backend = PredictingBackend()
    context = make_context(tmp_path)
    context.images[0].storage_path = "../escape.png"
    assert parsed(processor(tmp_path, backend).process(context)).images[0].status == "failed"
    context = make_context(tmp_path, max_pixels=100)
    assert parsed(processor(tmp_path, backend).process(context)).images[0].status == "failed"
    assert backend.calls == []


def test_mutated_box_instance_cannot_bypass_validation_or_abort_other_images(tmp_path):
    context = make_context(tmp_path, [image_bytes(), image_bytes()])
    boxes = dict(context.segmentation_boxes)
    boxes[context.images[0].asset_id] = BOX.model_copy(update={"x0": 2.0})
    backend = PredictingBackend()
    result = processor(tmp_path, backend).process(replace(context, segmentation_boxes=boxes))
    data = parsed(result)
    assert result.status == "partial"
    assert [image.status for image in data.images] == ["failed", "success"]
    assert data.images[0].prompt_box is None
    assert len(backend.calls) == 1


def test_real_inspection_receipt_is_reused_but_forged_or_stale_data_is_not(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    asset = context.images[0]
    path = context.resolve_asset(asset)
    calls = []
    inspect = image_metadata.inspect_image

    def record(source, max_pixels=40_000_000):
        calls.append(source)
        return inspect(source, max_pixels)

    monkeypatch.setattr(image_metadata, "inspect_image", record)
    receipt = image_metadata.InspectedImage(path)
    trusted = replace(context, inspected_images={asset.asset_id: receipt})
    assert parsed(processor(tmp_path, PredictingBackend()).process(trusted)).success_count == 1
    assert len(calls) == 1
    path.write_bytes(b"broken after inspection")
    assert parsed(processor(tmp_path, PredictingBackend()).process(trusted)).failed_count == 1
    forged = replace(context, inspected_images={asset.asset_id: {"width": 32, "height": 24}})
    assert parsed(processor(tmp_path, PredictingBackend()).process(forged)).failed_count == 1
    assert len(calls) == 3


def test_exif_rotation_converts_normalized_box_to_the_actual_model_frame(tmp_path):
    context = make_context(tmp_path, [image_bytes((12, 8), orientation=6)])
    backend = PredictingBackend()
    result = processor(tmp_path, backend).process(context)
    image = parsed(result).images[0]
    assert (image.width, image.height) == (8, 12)
    assert backend.calls == [((8, 12), (2.0, 3.0, 6.0, 9.0))]
    assert image.bbox == (2, 3, 6, 9)
    with Image.open(storage_io_path(tmp_path / image.mask_path)) as mask:
        assert mask.size == (8, 12)


def test_missing_transaction_storage_does_not_write_to_final_directory(tmp_path):
    context = replace(make_context(tmp_path), artifact_store=None)
    backend = PredictingBackend()
    result = processor(tmp_path, backend).process(context)
    assert parsed(result).images[0].status == "unavailable"
    assert backend.calls == []
    assert list(tmp_path.rglob("masks")) == []


def test_artifact_save_failure_is_a_per_image_failure(tmp_path):
    context = make_context(tmp_path)

    class FailingStore:
        def save_mask(self, asset_id, image):
            raise OSError("PRIVATE_STORAGE_PATH_C:\\private")

    result = processor(tmp_path, PredictingBackend()).process(replace(context, artifact_store=FailingStore()))
    assert parsed(result).images[0].status == "failed"
    assert "PRIVATE_STORAGE_PATH" not in result.model_dump_json()


def valid_record():
    return dict(
        asset_id=uuid4(), filename="image.png", status="success", prompt_type="box", prompt_box=BOX,
        width=32, height=24, bbox=(8, 6, 24, 18), mask_path=f"objects/{uuid4()}/masks/mask.png",
        area_ratio=0.25, confidence=0.95, warnings=["请人工检查目标。"],
    )


@pytest.mark.parametrize("changes", [
    {"x0": -0.1}, {"x1": 1.1}, {"x0": 0.8}, {"y1": 0.25}, {"x0": True}, {"x0": "0.1"}, {"x0": float("nan")},
])
def test_normalized_box_rejects_invalid_or_non_numeric_edges(changes):
    with pytest.raises(ValidationError):
        NormalizedBox.model_validate({**BOX.model_dump(), **changes})


@pytest.mark.parametrize("changes", [
    {"bbox": (-1, 0, 10, 10)}, {"bbox": (0, 0, 33, 24)}, {"bbox": (8, 6, 8, 18)},
    {"bbox": (True, 6, 24, 18)}, {"bbox": (8.0, 6, 24, 18)}, {"area_ratio": 1.1},
    {"confidence": float("nan")}, {"mask_path": "../mask.png"}, {"mask_path": "objects/a/masks/mask.jpg"},
    {"mask_path": None}, {"area_ratio": 0}, {"area_ratio": 1}, {"error_id": uuid4()},
    {"status": "disabled"}, {"prompt_type": "none"}, {"width": None},
])
def test_segmentation_record_rejects_invalid_geometry_paths_and_states(changes):
    with pytest.raises(ValidationError):
        SegmentationImage.model_validate({**valid_record(), **changes})


@pytest.mark.parametrize("status", ["disabled", "unavailable", "needs_input", "failed"])
def test_inactive_states_cannot_claim_area_or_confidence(status):
    fields = dict(asset_id=uuid4(), filename="image.png", status=status, error_id=uuid4())
    for field in ("area_ratio", "confidence"):
        with pytest.raises(ValidationError):
            SegmentationImage.model_validate({**fields, field: 0.5})


def test_batch_counts_and_asset_ids_are_consistent():
    record = SegmentationImage.model_validate(valid_record())
    valid = dict(input_count=1, success_count=1, review_count=0, failed_count=0, skipped_count=0, images=[record])
    assert SegmentationData.model_validate(valid).input_count == 1
    for change in ({"success_count": 0}, {"input_count": 2}, {"success_count": True}, {"skipped_count": -1}):
        with pytest.raises(ValidationError):
            SegmentationData.model_validate({**valid, **change})
    with pytest.raises(ValidationError):
        SegmentationData.model_validate({**valid, "input_count": 2, "success_count": 2, "images": [record, record]})


def test_empty_batch_is_a_valid_skipped_result(tmp_path):
    context = ProcessingContext(images=(), data_dir=tmp_path)
    result = ObjectSegmentationProcessor(Settings(data_dir=tmp_path)).process(context)
    assert result.status == "skipped"
    assert parsed(result).input_count == 0


def test_api_constructor_is_lazy_and_missing_credentials_never_open_network(tmp_path, monkeypatch):
    import urllib.request

    calls = []

    def denied_open(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Unit tests must never make a cloud request")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", denied_open)
    backend = BaiduSegmentationBackend("", "")
    with Image.new("RGB", (128, 128)) as image:
        with pytest.raises(SegmentationUnavailable, match="API Key"):
            backend.predict(image, (16, 16, 112, 112))
    assert calls == []


def test_default_processor_uses_cloud_backend_without_requests_at_initialization(tmp_path, monkeypatch):
    import urllib.request

    def denied_open(*args, **kwargs):
        raise AssertionError("Creating or disabling the processor must not call the service")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", denied_open)
    configured = Settings(
        data_dir=tmp_path, segmentation_enabled=True,
        segmentation_api_key="unit-api-key", segmentation_secret_key="unit-secret-key",
    )
    current = ObjectSegmentationProcessor(configured)
    assert isinstance(current.backend, BaiduSegmentationBackend)
    assert parsed(current.process(make_context(tmp_path, boxes=False))).images[0].status == "needs_input"


def test_provider_processing_notes_are_retained_without_fabricating_confidence(tmp_path):
    class CloudLikePrediction(PredictingBackend):
        def predict(self, request):
            prediction = super().predict(request)
            return MaskPrediction(prediction.mask, None, warnings=("云端分析经过缩放，边界需要人工复查。",))

    result = processor(tmp_path, CloudLikePrediction()).process(make_context(tmp_path))
    item = parsed(result).images[0]
    assert item.model_name == "baidu_intelligent_segmentation"
    assert item.status == "needs_confirmation"
    assert item.confidence is None
    assert "云端分析经过缩放，边界需要人工复查。" in item.warnings
    assert item.mask_path


@pytest.mark.parametrize("model_name", ["baidu_intelligent_segmentation", "sam2.1_hiera_tiny"])
def test_current_cloud_and_legacy_local_model_identifiers_remain_readable(model_name):
    record = SegmentationImage.model_validate({**valid_record(), "model_name": model_name})
    assert record.model_name == model_name


@pytest.mark.skipif(
    os.environ.get("DIGITIZER_RUN_SEGMENTATION_API_SMOKE") != "1",
    reason="Explicit opt-in required: uploads a synthetic image to Baidu and may incur API charges",
)
def test_real_segmentation_api_smoke_requires_explicit_opt_in():
    settings = Settings()
    backend = BaiduSegmentationBackend(
        settings.segmentation_api_key, settings.segmentation_secret_key, settings.segmentation_timeout_seconds,
    )
    with Image.new("RGB", (256, 192), "white") as image:
        ImageDraw.Draw(image).ellipse((64, 48, 191, 143), fill="blue")
        prediction = backend.predict(image, (32, 24, 224, 168))
    with prediction.mask as mask:
        assert mask.mode == "L" and mask.size == (256, 192)
        assert {index for index, count in enumerate(mask.histogram()) if count}.issubset({0, 255})
    assert prediction.score is None
