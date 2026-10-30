"""Mask contract tests use injected predictions; real SAM2 smoke is opt-in."""

from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

from PIL import Image, ImageDraw
from pydantic import ValidationError
import pytest

from app.config import Settings
from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQuality, ImageQualityData
from app.models.processing_result import ProcessingResult
from app.models.segmentation import NormalizedBox, SegmentationData, SegmentationImage
from app.modules import image_metadata, sam2_backend
from app.modules.base import ProcessingContext
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.modules.sam2_backend import MaskPrediction, Sam2Backend, SegmentationUnavailable


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
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask.save(destination, format="PNG")
        self.paths.append(path)
        return path


class PredictingBackend:
    def __init__(self, outcome="normal", score=0.95):
        self.outcome, self.score, self.calls = outcome, score, []

    def predict(self, image, box):
        self.calls.append((image.size, box))
        if self.outcome == "error" or (self.outcome == "second_error" and len(self.calls) == 2):
            raise RuntimeError("PRIVATE_MODEL_PATH_C:\\private\\weights")
        if self.outcome == "unavailable":
            raise SegmentationUnavailable("分割模型无法加载，请检查可选依赖。")
        mask = Image.new("L", image.size, 255 if self.outcome == "full" else 0)
        if self.outcome not in {"empty", "full"}:
            x0, y0, x1, y1 = box
            ImageDraw.Draw(mask).rectangle((int(x0), int(y0), int(x1) - 1, int(y1) - 1), fill=255)
        if self.outcome == "gray":
            mask.putpixel((0, 0), 128)
        if self.outcome == "wrong_size":
            mask = mask.resize((3, 3))
        return MaskPrediction(mask, self.score)


def make_context(tmp_path, contents=None, *, boxes=True, max_pixels=40_000_000):
    object_id = uuid4()
    contents = contents or [image_bytes()]
    assets = []
    for index, content in enumerate(contents):
        asset_id = uuid4()
        relative = f"objects/{object_id}/images/{asset_id}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
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
    assert "正确率" in item.warnings[0]
    assert "mask" not in result.data  # No pixel arrays are embedded in JSON.
    with Image.open(tmp_path / item.mask_path) as saved:
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
    disabled = ObjectSegmentationProcessor(Settings(data_dir=tmp_path), backend=backend)
    result = disabled.process(make_context(tmp_path))
    assert result.status == "skipped"
    assert parsed(result).images[0].status == "disabled"
    result = processor(tmp_path, backend).process(make_context(tmp_path, boxes=False))
    assert result.status == "needs_input"
    assert parsed(result).images[0].status == "needs_input"
    assert backend.calls == []


def test_missing_checkpoint_and_dependency_or_load_failure_are_unavailable(tmp_path):
    context = make_context(tmp_path)
    missing = ObjectSegmentationProcessor(Settings(data_dir=tmp_path, segmentation_enabled=True))
    result = missing.process(context)
    assert result.status == "skipped"
    assert parsed(result).images[0].status == "unavailable"
    assert "权重" in result.errors[0].message
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
    with Image.open(tmp_path / image.mask_path) as mask:
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


def test_backend_constructor_is_lazy_and_missing_weights_do_not_import_optional_packages(tmp_path, monkeypatch):
    import builtins

    original_import = builtins.__import__
    imported = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "sam2"}:
            imported.append(name)
            raise AssertionError("optional imports must remain lazy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    backend = Sam2Backend(tmp_path / "missing.pt")
    with pytest.raises(SegmentationUnavailable, match="权重"):
        backend.predict(Image.new("RGB", (8, 8)), (1, 1, 7, 7))
    assert imported == []


def test_backend_loads_official_api_once_and_disables_optional_cuda_postprocessing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.write_bytes(b"unit test does not contain model weights")
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    build_module, predictor_module = ModuleType("sam2.build_sam"), ModuleType("sam2.sam2_image_predictor")

    def build(config, path, **kwargs):
        calls.append((config, path, kwargs))
        return object()

    class Predictor:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs

    build_module.build_sam2 = build
    predictor_module.SAM2ImagePredictor = Predictor
    monkeypatch.setitem(sys.modules, "sam2", ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_module)
    backend = Sam2Backend(checkpoint)
    first = backend._load()
    assert backend._load() is first
    assert len(calls) == 1
    assert calls[0] == ("configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), {"device": "cpu", "apply_postprocessing": False})
    assert first.kwargs == {"max_hole_area": 0, "max_sprinkle_area": 0}
    with pytest.raises(SegmentationUnavailable, match="CUDA"):
        Sam2Backend(checkpoint, "cuda")._load()


def test_dependency_import_failure_is_safe_and_cached_until_restart(tmp_path, monkeypatch):
    import builtins

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    real_import = builtins.__import__
    attempted = []

    def missing_numpy(name, *args, **kwargs):
        if name == "numpy":
            attempted.append(name)
            raise ModuleNotFoundError("PRIVATE_PACKAGE_PATH_C:\\private\\numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_numpy)
    backend = Sam2Backend(checkpoint)
    for _ in range(2):
        with pytest.raises(SegmentationUnavailable) as error:
            backend._load()
        assert "PRIVATE_PACKAGE_PATH" not in str(error.value)
        assert "可选依赖" in str(error.value)
    assert attempted == ["numpy"]


def test_checkpoint_builder_failure_is_safe_and_not_reinitialized_per_image(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    build_module, predictor_module = ModuleType("sam2.build_sam"), ModuleType("sam2.sam2_image_predictor")
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("PRIVATE_WEIGHT_PATH_C:\\private\\weights")

    build_module.build_sam2 = fail
    predictor_module.SAM2ImagePredictor = object
    monkeypatch.setitem(sys.modules, "sam2", ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_module)
    backend = Sam2Backend(checkpoint)
    for _ in range(2):
        with pytest.raises(SegmentationUnavailable) as error:
            backend._load()
        assert "PRIVATE_WEIGHT_PATH" not in str(error.value)
        assert "模型加载失败" in str(error.value)
    assert len(calls) == 1


@pytest.mark.skipif(os.environ.get("DIGITIZER_RUN_SAM2_SMOKE") != "1", reason="Real SAM2 inference needs explicit opt-in, optional dependencies and a local checkpoint")
def test_real_sam2_checkpoint_smoke_without_any_automatic_download():
    settings = Settings()
    with Image.new("RGB", (128, 96), "white") as image:
        ImageDraw.Draw(image).ellipse((32, 24, 95, 71), fill="blue")
        prediction = Sam2Backend(settings.segmentation_checkpoint, settings.segmentation_device).predict(image, (24, 16, 104, 80))
    with prediction.mask as mask:
        assert mask.mode == "L" and mask.size == (128, 96)
        assert {index for index, count in enumerate(mask.histogram()) if count}.issubset({0, 255})
