"""Pillow quality screening, trusted validation reuse, and typed result boundaries."""

from dataclasses import replace
from io import BytesIO
from uuid import uuid4

from PIL import Image, ImageDraw
from pydantic import ValidationError
import pytest

from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQuality, ImageQualityData
from app.modules import image_metadata, image_quality
from app.modules.base import ProcessingContext
from app.modules.image_metadata import InspectedImage
from app.modules.image_quality import DEFAULT_THRESHOLDS, ImageQualityProcessor, QualityThresholds


def checker(size=(960, 720), values=(64, 192), tile=8):
    image = Image.new("L", size, values[0])
    drawing = ImageDraw.Draw(image)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2:
                drawing.rectangle((x, y, x + tile - 1, y + tile - 1), fill=values[1])
    return image


def encoded(image, image_format="PNG", **kwargs):
    buffer = BytesIO()
    image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


def context_with(tmp_path, contents, *, max_pixels=40_000_000):
    object_id = uuid4()
    assets = []
    for content in contents:
        asset_id = uuid4()
        relative = f"objects/{object_id}/images/{asset_id}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        # These deliberately inaccurate hints cannot replace real-file inspection.
        assets.append(ImageAsset(
            asset_id=asset_id, filename="same-name.png", storage_path=relative,
            width=1, height=1, mime_type="image/png", image_format="PNG", size_bytes=1,
        ))
    return ProcessingContext(images=tuple(assets), data_dir=tmp_path, max_image_pixels=max_pixels)


def assess(tmp_path, image):
    context = context_with(tmp_path, [encoded(image)])
    result = ImageQualityProcessor().process(context)
    assert result.status == "success", result.model_dump()
    payload = ImageQualityData.model_validate(result.data)
    return context, result, payload.images[0]


def test_detailed_well_lit_image_has_real_measured_metrics_and_good_status(tmp_path):
    _, result, image = assess(tmp_path, checker())
    assert image.status == "good"
    assert image.issues == image.suggestions == []
    assert image.width == 960 and image.height == 720
    assert image.mean_luminance == pytest.approx(128)
    assert image.contrast_stddev == pytest.approx(64)
    assert image.edge_strength > DEFAULT_THRESHOLDS.low_edge_strength
    assert result.warnings == []
    assert "启发式" in result.data["assessment_note"]
    assert "不保证" in result.data["assessment_note"]


@pytest.mark.parametrize("image, issue", [
    (checker((128, 96)), "low_resolution"),
    (Image.new("L", (640, 480), 10), "too_dark"),
    (Image.new("L", (640, 480), 245), "too_bright"),
    (checker((640, 480), (118, 138)), "low_contrast"),
    (Image.new("L", (640, 480), 128), "low_detail"),
])
def test_screening_issues_have_chinese_suggestions_without_rejecting_image(tmp_path, image, issue):
    _, result, quality = assess(tmp_path, image)
    assert quality.status == "review"
    assert issue in quality.issues
    assert len(quality.suggestions) == len(quality.issues)
    assert all(any("\u4e00" <= char <= "\u9fff" for char in suggestion) for suggestion in quality.suggestions)
    assert result.warnings and result.errors == []


def test_high_contrast_smooth_gradient_is_reviewed_for_detail_without_blur_diagnosis(tmp_path):
    image = Image.new("L", (640, 480))
    image.putdata([30 + int(190 * x / 639) for _ in range(480) for x in range(640)])
    _, _, quality = assess(tmp_path, image)
    assert quality.issues == ["low_detail"]
    assert quality.contrast_stddev > DEFAULT_THRESHOLDS.low_contrast_stddev
    assert "平滑表面" in quality.suggestions[0]
    assert "人工" in quality.suggestions[0]
    assert "blur" not in quality.issues


@pytest.mark.parametrize("content", [b"broken", encoded(Image.new("RGB", (32, 24)), "BMP")])
def test_corrupt_or_unsupported_image_has_safe_failed_result(tmp_path, content):
    context = context_with(tmp_path, [content])
    result = ImageQualityProcessor().process(context)
    payload = ImageQualityData.model_validate(result.data)
    assert result.status == "failed"
    assert (payload.input_count, payload.checked_count, payload.review_count) == (1, 0, 0)
    assert result.errors[0].asset_id == context.images[0].asset_id
    assert result.errors[0].error_id is not None
    assert str(tmp_path) not in result.model_dump_json()


def test_readable_jpeg_header_does_not_allow_truncated_pixels(tmp_path):
    content = encoded(Image.new("RGB", (100, 100), "gray"), "JPEG")[:-20]
    with Image.open(BytesIO(content)) as header:
        assert header.size == (100, 100)
    result = ImageQualityProcessor().process(context_with(tmp_path, [content]))
    assert result.status == "failed"


def test_missing_file_is_isolated_and_other_images_keep_their_asset_ids(tmp_path):
    context = context_with(tmp_path, [encoded(checker()), encoded(checker()), encoded(Image.new("L", (640, 480), 10))])
    context.resolve_asset(context.images[1]).unlink()
    result = ImageQualityProcessor().process(context)
    payload = ImageQualityData.model_validate(result.data)
    assert result.status == "partial"
    assert (payload.input_count, payload.checked_count, payload.review_count) == (3, 2, 1)
    assert [image.asset_id for image in payload.images] == [context.images[0].asset_id, context.images[2].asset_id]
    assert len({image.filename for image in payload.images}) == 1
    assert result.errors[0].asset_id == context.images[1].asset_id


def test_actual_pixel_limit_is_enforced_in_standalone_processor(tmp_path):
    context = context_with(tmp_path, [encoded(checker())], max_pixels=100)
    result = ImageQualityProcessor().process(context)
    assert result.status == "failed"
    assert result.data["checked_count"] == 0


def test_valid_receipt_skips_duplicate_verify_but_still_computes_quality(tmp_path, monkeypatch):
    context = context_with(tmp_path, [encoded(checker())])
    asset = context.images[0]
    receipt = InspectedImage(context.resolve_asset(asset))
    context = replace(context, inspected_images={asset.asset_id: receipt})

    def unwanted_inspection(*args, **kwargs):
        raise AssertionError("A valid unchanged receipt should avoid another full validation")

    monkeypatch.setattr(image_metadata, "inspect_image", unwanted_inspection)
    result = ImageQualityProcessor().process(context)
    assert result.status == "success"
    assert result.data["images"][0]["contrast_stddev"] == pytest.approx(64)


def test_forged_or_stale_receipt_cannot_skip_file_validation(tmp_path):
    context = context_with(tmp_path, [encoded(checker())])
    asset = context.images[0]
    receipt = InspectedImage(context.resolve_asset(asset))
    context.resolve_asset(asset).write_bytes(b"broken")
    for untrusted in (receipt, {"width": 960, "height": 720}):
        result = ImageQualityProcessor().process(replace(context, inspected_images={asset.asset_id: untrusted}))
        assert result.status == "failed"


def test_tighter_pixel_limit_does_not_trust_an_old_receipt(tmp_path):
    context = context_with(tmp_path, [encoded(checker())])
    asset = context.images[0]
    receipt = InspectedImage(context.resolve_asset(asset))
    result = ImageQualityProcessor().process(replace(context, max_image_pixels=100, inspected_images={asset.asset_id: receipt}))
    assert result.status == "failed"


def test_unsafe_path_is_rejected_before_opening_any_file(tmp_path, monkeypatch):
    context = context_with(tmp_path, [encoded(checker())])
    context.images[0].storage_path = "../outside.png"
    opened = []

    def unexpected_open(*args, **kwargs):
        opened.append(True)
        raise AssertionError("Path must be rejected first")

    monkeypatch.setattr(Image, "open", unexpected_open)
    result = ImageQualityProcessor().process(context)
    assert result.status == "failed"
    assert not opened


def test_analysis_is_bounded_and_original_bytes_are_unchanged(tmp_path, monkeypatch):
    context = context_with(tmp_path, [encoded(checker((3000, 1500)))])
    path = context.resolve_asset(context.images[0])
    original = path.read_bytes()
    sizes = []
    original_edge = image_quality._edge_strength

    def record_size(gray):
        sizes.append(gray.size)
        return original_edge(gray)

    monkeypatch.setattr(image_quality, "_edge_strength", record_size)
    result = ImageQualityProcessor().process(context)
    quality = ImageQualityData.model_validate(result.data).images[0]
    assert (quality.width, quality.height) == (3000, 1500)
    assert sizes == [(1024, 512)]
    assert (quality.analysis_width, quality.analysis_height) == sizes[0]
    assert path.read_bytes() == original


def test_exif_rotation_is_applied_after_shrinking_without_changing_original_dimensions(tmp_path):
    exif = Image.Exif()
    exif[274] = 6
    content = encoded(checker((1200, 600)), "JPEG", exif=exif)
    context = context_with(tmp_path, [content])
    result = ImageQualityProcessor().process(context)
    quality = ImageQualityData.model_validate(result.data).images[0]
    assert (quality.width, quality.height) == (1200, 600)
    assert (quality.analysis_width, quality.analysis_height) == (512, 1024)
    assert context.resolve_asset(context.images[0]).read_bytes() == content


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparent_dark_pixels_are_measured_on_white_background(tmp_path, mode):
    if mode == "P":
        image = Image.new("P", (640, 480), 0)
        image.putpalette([0, 0, 0] * 256)
        image.info["transparency"] = 0
    else:
        image = Image.new(mode, (640, 480), (0, 0, 0, 0) if mode == "RGBA" else (0, 0))
    _, _, quality = assess(tmp_path, image)
    assert quality.mean_luminance == pytest.approx(255)
    assert "too_dark" not in quality.issues


def test_sixteen_bit_grayscale_is_scaled_instead_of_clipped(tmp_path):
    _, _, quality = assess(tmp_path, Image.new("I;16", (640, 480), 32768))
    assert quality.mean_luminance == pytest.approx(127, abs=1)
    assert "too_dark" not in quality.issues and "too_bright" not in quality.issues


@pytest.mark.parametrize("transparent_value", [0, 65535])
def test_sixteen_bit_grayscale_with_transparent_key_fails_instead_of_reporting_false_exposure(tmp_path, transparent_value):
    content = encoded(Image.new("I;16", (640, 480), 32768), transparency=transparent_value)
    result = ImageQualityProcessor().process(context_with(tmp_path, [content]))
    assert result.status == "failed"
    assert result.data["images"] == []
    assert "16位灰度" in result.errors[0].message
    assert "8位PNG或JPEG" in result.errors[0].message


def test_unsupported_transparent_grayscale_does_not_discard_other_quality_results(tmp_path):
    content = encoded(Image.new("I;16", (640, 480), 32768), transparency=0)
    context = context_with(tmp_path, [encoded(checker()), content])
    result = ImageQualityProcessor().process(context)
    payload = ImageQualityData.model_validate(result.data)
    assert result.status == "partial"
    assert (payload.input_count, payload.checked_count, payload.review_count) == (2, 1, 0)
    assert payload.images[0].asset_id == context.images[0].asset_id
    assert result.errors[0].asset_id == context.images[1].asset_id


def test_single_pixel_detail_is_zero_without_crop_errors(tmp_path):
    _, _, quality = assess(tmp_path, Image.new("L", (1, 1), 128))
    assert quality.edge_strength == 0
    assert "low_resolution" in quality.issues


def valid_quality():
    return ImageQuality(
        asset_id=uuid4(), filename="photo.png", width=640, height=480,
        analysis_width=640, analysis_height=480, mean_luminance=128,
        contrast_stddev=64, edge_strength=16, status="good",
    ).model_dump(mode="json")


@pytest.mark.parametrize("changes", [
    {"mean_luminance": float("nan")}, {"mean_luminance": 256}, {"contrast_stddev": 128},
    {"edge_strength": -1}, {"status": "blurry"}, {"width": 0}, {"analysis_width": 2049},
    {"status": "review"}, {"issues": ["low_detail"], "suggestions": ["请查看细节"]},
    {"status": "review", "issues": ["low_detail"], "suggestions": []},
    {"status": "review", "issues": ["low_detail", "low_detail"], "suggestions": ["查看", "查看"]},
    {"status": "review", "issues": ["low_detail"], "suggestions": [""]},
    {"filename": "bad\x00name"},
])
def test_image_quality_model_rejects_invalid_metrics_or_inconsistent_findings(changes):
    with pytest.raises(ValidationError):
        ImageQuality.model_validate({**valid_quality(), **changes})


@pytest.mark.parametrize("changes", [
    {"input_count": 0}, {"checked_count": 0}, {"review_count": 1}, {"input_count": -1},
    {"checked_count": True}, {"images": []}, {"method": "reliable_blur_diagnosis"},
])
def test_batch_model_rejects_inconsistent_counts(changes):
    data = {"input_count": 1, "checked_count": 1, "review_count": 0, "images": [valid_quality()]}
    with pytest.raises(ValidationError):
        ImageQualityData.model_validate({**data, **changes})


def test_batch_model_rejects_duplicate_asset_ids():
    image = valid_quality()
    with pytest.raises(ValidationError):
        ImageQualityData(input_count=2, checked_count=2, review_count=0, images=[image, image])


@pytest.mark.parametrize("changes", [
    {"max_analysis_side": 1}, {"max_analysis_side": 4096}, {"dark_mean": float("nan")},
    {"dark_mean": 220, "bright_mean": 210}, {"min_pixels": 0}, {"min_short_side": True},
    {"low_contrast_stddev": -1}, {"low_edge_strength": float("inf")},
])
def test_threshold_configuration_cannot_disable_resource_or_metric_bounds(changes):
    with pytest.raises(ValueError):
        QualityThresholds(**changes)


def test_empty_input_has_valid_skipped_result(tmp_path):
    result = ImageQualityProcessor().process(context_with(tmp_path, []))
    assert result.status == "skipped"
    payload = ImageQualityData.model_validate(result.data)
    assert (payload.input_count, payload.checked_count, payload.review_count) == (0, 0, 0)
