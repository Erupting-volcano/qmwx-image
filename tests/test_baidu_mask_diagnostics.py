"""Offline Baidu boundary diagnostics; no assumptions about the real failed PNG."""

import base64
from io import BytesIO
import json
import logging
from uuid import UUID

from PIL import Image, ImageDraw
import pytest

from app.modules.baidu_backend import BaiduSegmentationBackend
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.models.segmentation import NormalizedBox, SegmentationData
from app.services.image_collection import ImageCollectionService, ImageInput


SIZE = (128, 160)


def encoded(image, fmt="PNG", **kwargs):
    with BytesIO() as stream:
        image.save(stream, format=fmt, **kwargs)
        return base64.b64encode(stream.getvalue()).decode("ascii")


def events(caplog, name="baidu_mask_decode"):
    return [json.loads(record.getMessage().split(" ", 1)[1]) for record in caplog.records
            if record.name == "app.modules.baidu_backend" and record.getMessage().startswith(name + " ")]


@pytest.mark.parametrize("mode", ["L", "1"])
@pytest.mark.parametrize("fill", [0, 255])
def test_binary_png_decodes_and_reports_safe_header_and_load_status(caplog, mode, fill):
    caplog.set_level(logging.INFO)
    with Image.new(mode, SIZE, fill) as image:
        data = encoded(image)
    result = BaiduSegmentationBackend._decode_mask(data, SIZE, [])
    with result:
        assert result.mode == "L" and result.size == SIZE and result.getextrema() == (fill, fill)
    entry = events(caplog)[0]
    assert entry["response_format"] == "PNG" and entry["response_mode"] == mode
    assert (entry["response_width"], entry["response_height"]) == SIZE
    assert entry["decode_status"] == "pixels_loaded" and entry["normalization_status"] == "binary_l"
    assert entry["decoded_byte_length"] > 0 and UUID(entry["diagnostic_id"])
    assert data not in caplog.text


@pytest.mark.parametrize("mode", ["P", "RGB", "RGBA"])
def test_other_valid_binary_encodings_are_explicitly_unconfirmed_not_silently_converted(caplog, mode):
    caplog.set_level(logging.INFO)
    with Image.new(mode, SIZE, 0) as image:
        if mode == "P":
            image.putpalette([v for i in range(256) for v in (i, i, i)])
            image.putpixel((20, 30), 255)
        else:
            image.putpixel((20, 30), (255, 255, 255) if mode == "RGB" else (255, 255, 255, 255))
        data = encoded(image)
    with pytest.raises(Exception, match="编码模式暂不支持"):
        BaiduSegmentationBackend._decode_mask(data, SIZE, [])
    entry = events(caplog)[0]
    assert entry["response_mode"] == mode and entry["normalization_status"] == "rejected_mode"
    assert entry["decode_status"] == "header_read"  # Never claim pixels loaded.


@pytest.mark.parametrize("value,decode_status", [(None, "invalid_image_field"), ("", "invalid_image_field"),
                                                 ({"unexpected": "object"}, "invalid_image_field"),
                                                 ("INVALID_BASE64_SECRET!", "invalid_base64")])
def test_missing_invalid_or_corrupt_base64_has_a_distinct_safe_stage(caplog, value, decode_status):
    caplog.set_level(logging.INFO)
    with pytest.raises(Exception):
        BaiduSegmentationBackend._decode_mask(value, SIZE, [])
    assert events(caplog)[0]["decode_status"] == decode_status
    assert "INVALID_BASE64_SECRET" not in caplog.text


def test_unidentified_bytes_are_not_confused_with_png_mode_or_dimensions(caplog):
    caplog.set_level(logging.INFO)
    data = base64.b64encode(b"PRIVATE_RESPONSE_NOT_AN_IMAGE").decode()
    with pytest.raises(Exception, match="无法完整解码"):
        BaiduSegmentationBackend._decode_mask(data, SIZE, [])
    entry = events(caplog)[0]
    assert entry["decode_status"] == "image_decode_failed" and entry["response_mode"] is None
    assert "PRIVATE_RESPONSE" not in caplog.text and data not in caplog.text


@pytest.mark.parametrize("fmt", ["JPEG", "BMP"])
def test_non_png_rejected_with_actual_format_only_in_server_diagnostics(caplog, fmt):
    caplog.set_level(logging.INFO)
    with Image.new("L", SIZE, 255) as image:
        with pytest.raises(Exception, match="不是PNG"):
            BaiduSegmentationBackend._decode_mask(encoded(image, fmt), SIZE, [])
    entry = events(caplog)[0]
    assert entry["response_format"] == fmt and entry["normalization_status"] == "rejected_format"


def test_size_mismatch_never_loads_or_resizes_and_is_not_reported_as_mode_failure(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    with Image.new("L", (129, 160)) as image:
        data = encoded(image)
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong API dimensions must not reach pixel loading/resizing")
    monkeypatch.setattr(Image.Image, "resize", forbidden)
    with pytest.raises(Exception, match="尺寸.*不一致") as caught:
        BaiduSegmentationBackend._decode_mask(data, SIZE, [])
    entry = events(caplog)[0]
    assert entry["normalization_status"] == "rejected_dimensions" and entry["decode_status"] == "header_read"
    assert entry["response_width"] == 129 and entry["expected_width"] == 128
    assert "129" not in str(caught.value) and "response_width" not in str(caught.value)


def test_full_pixel_decode_failure_is_distinct_from_header_rejection(caplog):
    caplog.set_level(logging.INFO)
    with Image.new("L", SIZE) as image:
        raw = base64.b64decode(encoded(image))[:50]  # Header survives, IDAT is truncated.
    with pytest.raises(Exception):
        BaiduSegmentationBackend._decode_mask(base64.b64encode(raw).decode(), SIZE, [])
    assert events(caplog)[0]["decode_status"] == "image_decode_failed"


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_ordinary_color_picture_is_never_converted_into_a_mask(caplog, mode):
    caplog.set_level(logging.INFO)
    with Image.new(mode, SIZE, (50, 110, 190) if mode == "RGB" else (50, 110, 190, 128)) as image:
        with pytest.raises(Exception, match="编码模式暂不支持"):
            BaiduSegmentationBackend._decode_mask(encoded(image), SIZE, [])
    assert events(caplog)[0]["normalization_status"] == "rejected_mode"


@pytest.mark.parametrize("mode", ["L", "1"])
def test_transparent_single_channel_png_is_refused_instead_of_losing_alpha(caplog, mode):
    caplog.set_level(logging.INFO)
    with Image.new(mode, SIZE, 255) as image:
        data = encoded(image, transparency=255 if mode == "L" else 1)
    with pytest.raises(Exception, match="透明信息"):
        BaiduSegmentationBackend._decode_mask(data, SIZE, [])
    entry = events(caplog)[0]
    assert entry["has_transparency"] and entry["normalization_status"] == "rejected_transparency"


def test_palette_transparency_and_white_alpha_are_not_misinterpreted(caplog):
    caplog.set_level(logging.INFO)
    with Image.new("P", SIZE) as image:
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
        image.putpixel((1, 1), 1)
        p = encoded(image, transparency=bytes([0, 255]))
    with Image.new("RGBA", SIZE, (255, 255, 255, 0)) as image:
        image.putpixel((1, 1), (255, 255, 255, 255))
        rgba = encoded(image)
    for data in (p, rgba):
        with pytest.raises(Exception, match="编码模式暂不支持"):
            BaiduSegmentationBackend._decode_mask(data, SIZE, [])
    assert all(entry["has_transparency"] for entry in events(caplog))


def test_accepted_mask_drops_png_orientation_and_other_metadata(tmp_path):
    with Image.new("L", SIZE) as image:
        image.putpixel((10, 15), 255)
        exif = Image.Exif()
        exif[274] = 6
        data = encoded(image, exif=exif)
    with BaiduSegmentationBackend._decode_mask(data, SIZE, []) as mask:
        assert mask.info == {} and mask.getbbox() == (10, 15, 11, 16)
        destination = tmp_path / "normalized.png"
        mask.save(destination)
    with Image.open(destination) as saved:
        assert saved.size == SIZE and saved.mode == "L"
        assert not saved.getexif() and "transparency" not in saved.info


class _Response(BytesIO):
    status = 200
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


class _Opener:
    def __init__(self, payload):
        self.payload, self.calls = payload, []
    def open(self, request, timeout):
        self.calls.append(request)
        return _Response({"access_token": "TEST_PRIVATE_TOKEN", "expires_in": 3600} if len(self.calls) == 1 else self.payload)


@pytest.mark.parametrize("orientation", [1, 6])
def test_exif_input_alignment_padding_mask_commit_and_preview_contract(settings, tmp_path, caplog, orientation):
    caplog.set_level(logging.INFO)
    with Image.new("RGB", (80, 160), (40, 110, 190)) as image, BytesIO() as stream:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(stream, format="JPEG", exif=exif)
        original = stream.getvalue()
    oriented = (80, 160) if orientation == 1 else (160, 80)
    prepared = (max(128, oriented[0]), max(128, oriented[1]))
    left, top = (prepared[0] - oriented[0]) // 2, (prepared[1] - oriented[1]) // 2
    bbox = (10, 20, 40, 60)
    with Image.new("L", prepared) as mask:
        ImageDraw.Draw(mask).rectangle((bbox[0] + left, bbox[1] + top, bbox[2] + left - 1, bbox[3] + top - 1), fill=255)
        opener = _Opener({"image": encoded(mask)})
    backend = BaiduSegmentationBackend("TEST_PRIVATE_KEY", "TEST_PRIVATE_SECRET", opener=opener)
    from dataclasses import replace
    config = replace(settings, segmentation_enabled=True)
    service = ImageCollectionService(config, processors=[ObjectSegmentationProcessor(config, backend=backend)])
    profile = service.collect("本地坐标回归", [ImageInput("photo.jpg", BytesIO(original), NormalizedBox(x0=0.1,y0=0.1,x1=0.9,y1=0.9))])
    record = SegmentationData.model_validate(profile.processing_results[0].data).images[0]
    assert record.bbox == bbox and (record.width, record.height) == oriented
    assert record.status == "needs_confirmation"
    stored = service.storage.data_dir / record.mask_path
    with Image.open(stored) as saved:
        assert saved.size == oriented and saved.getbbox() == bbox and saved.mode == "L"
    assert (service.storage.data_dir / profile.images[0].storage_path).read_bytes() == original
    request = json.loads(opener.calls[-1].data)
    with Image.open(BytesIO(base64.b64decode(request["image"]))) as sent:
        assert sent.size == prepared
    assert len(opener.calls) == 2  # One simulated auth + one simulated segment.
    decode, geometry = events(caplog)[0], events(caplog, "baidu_mask_geometry")
    assert decode["diagnostic_id"] == geometry[0]["diagnostic_id"] == geometry[1]["diagnostic_id"]
    assert geometry[-1]["mapping_status"] == "mapped" and (geometry[-1]["final_width"], geometry[-1]["final_height"]) == oriented
    public = profile.model_dump_json()
    assert "diagnostic_id" not in public and "response_mode" not in public
    for secret in ("TEST_PRIVATE_KEY", "TEST_PRIVATE_SECRET", "TEST_PRIVATE_TOKEN", request["image"]):
        assert secret not in caplog.text and secret not in public


def test_server_error_id_and_decode_diagnostics_are_linked_without_ui_details(settings, caplog):
    caplog.set_level(logging.INFO)
    with Image.new("RGB", SIZE, (40, 100, 180)) as mask:
        opener = _Opener({"image": encoded(mask)})
    backend = BaiduSegmentationBackend("TEST_PRIVATE_KEY", "TEST_PRIVATE_SECRET", opener=opener)
    from dataclasses import replace
    config = replace(settings, segmentation_enabled=True)
    with Image.new("RGB", SIZE) as image, BytesIO() as stream:
        image.save(stream, format="PNG")
        source = stream.getvalue()
    profile = ImageCollectionService(config, processors=[ObjectSegmentationProcessor(config, backend=backend)]).collect(
        "诊断关联", [ImageInput("photo.png", BytesIO(source), NormalizedBox(x0=0.1,y0=0.1,x1=0.9,y1=0.9))])
    result = profile.processing_results[0]
    diagnostic = events(caplog)[0]
    assert str(result.errors[0].error_id) in caplog.text
    assert diagnostic["diagnostic_id"] in caplog.text and "百度掩码服务端诊断编号" in caplog.text
    assert "response_mode" not in profile.model_dump_json() and diagnostic["diagnostic_id"] not in profile.model_dump_json()
    assert not list(config.data_dir.rglob("masks/*.png"))
    assert all(secret not in caplog.text for secret in ("TEST_PRIVATE_KEY", "TEST_PRIVATE_SECRET", "TEST_PRIVATE_TOKEN"))
