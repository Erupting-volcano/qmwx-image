"""Request/result contracts only; constructed PIL masks are not model inference."""

from dataclasses import asdict

from PIL import Image
import pytest

from app.modules.segmentation_backend import (
    AutomaticPrompt, BoxPrompt, MaskCandidate, MaskPrediction, MaskPrompt,
    PointPrompt, SegmentationCapabilities, SegmentationRequest, TextPrompt,
    UnsupportedPromptError, check_request_capabilities,
)


@pytest.fixture
def rgb():
    with Image.new("RGB", (40, 30)) as image:
        yield image


def test_request_has_explicit_oriented_pixel_coordinates(rgb):
    box = BoxPrompt((0, 1, 40, 30))
    request = SegmentationRequest(rgb, box)
    assert request.image is rgb and request.prompt is box
    assert request.coordinate_system == "exif_oriented_pixels"
    with pytest.raises(ValueError):
        SegmentationRequest(rgb, box, coordinate_system="normalized")


@pytest.mark.parametrize("box", [(-1, 0, 4, 4), (3, 0, 2, 4), (0, 1, 4, 1),
                                  (0, 0, float('nan'), 1), (0, 0, True, 1)])
def test_invalid_box_is_rejected(box):
    with pytest.raises(ValueError):
        BoxPrompt(box)


def test_box_cannot_exceed_image(rgb):
    with pytest.raises(ValueError):
        SegmentationRequest(rgb, BoxPrompt((0, 0, 41, 20)))


def test_points_require_matching_binary_labels_and_image_bounds(rgb):
    request = SegmentationRequest(rgb, PointPrompt(((0, 0), (39, 29)), (1, 0)))
    check_request_capabilities(request, SegmentationCapabilities(point_prompt=True))
    for points, labels in [(((1, 1),), ()), (((1, 1),), (True,)), (((1, 1),), (2,))]:
        with pytest.raises(ValueError):
            PointPrompt(points, labels)
    with pytest.raises(ValueError):
        SegmentationRequest(rgb, PointPrompt(((40, 2),), (1,)))


def test_mask_prompt_is_same_size_and_binary(rgb):
    with Image.new("L", rgb.size) as mask:
        request = SegmentationRequest(rgb, MaskPrompt(mask))
        check_request_capabilities(request, SegmentationCapabilities(mask_prompt=True))
        mask.putpixel((1, 1), 128)
        # Mutating a PIL object after constructing a frozen request cannot bypass validation.
        with pytest.raises(ValueError):
            check_request_capabilities(request, SegmentationCapabilities(mask_prompt=True))
    with Image.new("L", (2, 2)) as mask, pytest.raises(ValueError):
        SegmentationRequest(rgb, MaskPrompt(mask))


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_request_rejects_non_rgb_images(mode):
    with Image.new(mode, (10, 10)) as image, pytest.raises(ValueError):
        SegmentationRequest(image, AutomaticPrompt())


@pytest.mark.parametrize("prompt,field", [
    (BoxPrompt((1, 1, 10, 10)), "box_prompt"),
    (PointPrompt(((1, 1),), (1,)), "point_prompt"),
    (TextPrompt("牛奶盒"), "text_prompt"),
    (AutomaticPrompt(), "automatic_mask"),
])
def test_capabilities_do_not_coerce_one_prompt_into_another(rgb, prompt, field):
    request = SegmentationRequest(rgb, prompt)
    with pytest.raises(UnsupportedPromptError):
        check_request_capabilities(request, SegmentationCapabilities())
    check_request_capabilities(request, SegmentationCapabilities(**{field: True}))
    assert request.prompt is prompt


@pytest.mark.parametrize("text", ["", " ", "x\ny", "x" * 2049])
def test_text_prompt_has_bounded_clear_content(text):
    with pytest.raises(ValueError):
        TextPrompt(text)


def test_unknown_prompt_is_never_ignored(rgb):
    with pytest.raises(UnsupportedPromptError):
        SegmentationRequest(rgb, object())


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_capabilities_require_real_booleans(value):
    with pytest.raises(ValueError):
        SegmentationCapabilities(box_prompt=value)


def test_capabilities_have_no_implicit_defaults():
    assert not any(asdict(SegmentationCapabilities()).values())


def test_prediction_keeps_legacy_arguments_and_new_fields_default_empty():
    with Image.new("L", (4, 4)) as mask:
        prediction = MaskPrediction(mask, 0.8, ("test only",))
        assert prediction.mask is mask and prediction.score == 0.8
        assert prediction.candidates == () and prediction.score_type is None
        assert prediction.model_id is None and prediction.model_version is None
        assert prediction.processing_time is None and prediction.extensions == {}


def test_multiple_candidates_and_different_score_semantics_are_retained():
    with Image.new("L", (4, 4)) as main, Image.new("L", (4, 4)) as other:
        candidate = MaskCandidate(other, 12.3, "text_match_logit", ("人工确认",))
        result = MaskPrediction(main, 0.82, candidates=(candidate,), score_type="predicted_iou",
                                model_id="future_model", model_version="test", processing_time=0.01)
        assert result.score_type != result.candidates[0].score_type
        assert result.candidates[0].score == 12.3
        assert result.candidates[0].mask is other


def test_extensions_are_namespaced_small_json_and_detached_from_caller():
    with Image.new("L", (1, 1)) as mask:
        data = {"baidu": {"api_mode": "control", "sizes": [128, 128]}}
        result = MaskPrediction(mask, extensions=data)
        data["baidu"]["sizes"].append(256)
        assert result.extensions == {"baidu": {"api_mode": "control", "sizes": [128, 128]}}


@pytest.mark.parametrize("extensions", [
    {"bad namespace": {}}, {"baidu": {"raw": b"binary"}},
    {"baidu": {"raw": "x" * 20000}}, {"baidu": {"raw": float('nan')}},
])
def test_extensions_never_embed_binary_or_large_outputs(extensions):
    with Image.new("L", (1, 1)) as mask, pytest.raises(ValueError):
        MaskPrediction(mask, extensions=extensions)


@pytest.mark.parametrize("metadata", [
    {"processing_time": -1}, {"processing_time": float('inf')},
    {"model_id": "../unsafe"}, {"model_version": ""}, {"score_type": "a\nb"},
    {"candidates": (object(),)},
])
def test_prediction_rejects_invalid_additive_metadata(metadata):
    with Image.new("L", (1, 1)) as mask, pytest.raises(ValueError):
        MaskPrediction(mask, **metadata)
