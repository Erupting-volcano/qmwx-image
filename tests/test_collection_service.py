"""The parent project can collect images without invoking the HTTP adapter."""

from io import BytesIO
import json
from dataclasses import replace

import pytest

from app.services.image_collection import CollectionError, ImageCollectionService, ImageInput


def test_direct_collection_returns_persisted_contract(settings, png_bytes):
    service = ImageCollectionService(settings)
    stream = BytesIO(png_bytes)
    profile = service.collect("主项目图片输入", [ImageInput("front.png", stream)])
    persisted = settings.data_dir / "objects" / str(profile.object_id) / "profile.json"
    assert profile.model_dump(mode="json") == json.loads(persisted.read_text(encoding="utf-8"))
    assert service.get_profile(str(profile.object_id)) == profile
    assert profile.images[0].image_format == "PNG"
    assert profile.measurements[0].value is None
    assert not stream.closed  # Caller owns the stream lifecycle.


def test_direct_collection_invalid_batch_is_rolled_back(settings, png_bytes):
    service = ImageCollectionService(settings)
    with pytest.raises(CollectionError) as rejected:
        service.collect("无效输入", [ImageInput("front.png", BytesIO(png_bytes)),
                                    ImageInput("bad.png", BytesIO(b"broken"))])
    assert rejected.value.code == "invalid_image"
    assert not list(settings.data_dir.rglob("profile.json"))
    assert not list(settings.data_dir.rglob("*.png"))


def test_direct_collection_input_limits_apply_without_http(settings, png_bytes):
    service = ImageCollectionService(settings, processors=[])
    with pytest.raises(CollectionError):
        service.collect("", [ImageInput("front.png", BytesIO(png_bytes))])
    with pytest.raises(CollectionError):
        service.collect("空输入", [])
    profile = service.collect("仅采集", [ImageInput("front.png", BytesIO(png_bytes))])
    assert profile.processing_results == []
    assert profile.status == "completed_with_warnings"


@pytest.mark.parametrize("name", [None, "", "   ", "a" * 121, "bad\x00name"])
def test_direct_collection_rejects_invalid_names(settings, png_bytes, name):
    stream = BytesIO(png_bytes)
    with pytest.raises(CollectionError) as rejected:
        ImageCollectionService(settings).collect(name, [ImageInput("front.png", stream)])
    assert rejected.value.code == "invalid_name"
    assert not stream.closed
    assert not settings.data_dir.exists()


@pytest.mark.parametrize("limit", ["max_image_bytes", "max_total_bytes", "max_image_pixels", "max_files"])
def test_direct_collection_enforces_limits_without_request_parser(settings, png_bytes, limit):
    values = {"max_image_bytes": len(png_bytes) - 1, "max_total_bytes": len(png_bytes) * 2 - 1,
              "max_image_pixels": 32 * 24 - 1, "max_files": 1}
    settings = replace(settings, **{limit: values[limit]})
    streams = [BytesIO(png_bytes) for _ in range(2 if limit in {"max_total_bytes", "max_files"} else 1)]
    with pytest.raises(CollectionError) as rejected:
        ImageCollectionService(settings).collect("限制检查", [ImageInput("front.png", stream) for stream in streams])
    expected = "invalid_images" if limit == "max_files" else "upload_too_large" if limit == "max_total_bytes" else "image_too_large"
    assert rejected.value.code == expected
    assert rejected.value.status_code == (422 if limit == "max_files" else 413)
    assert all(not stream.closed for stream in streams)
    assert not list(settings.data_dir.rglob("profile.json"))
    staging = settings.data_dir / ".staging"
    assert not staging.exists() or not list(staging.iterdir())


@pytest.mark.parametrize("failure", ["read_error", "text_chunk"])
def test_bad_stream_after_partial_read_is_rolled_back_without_closing_input(settings, png_bytes, failure):
    class FailingStream(BytesIO):
        calls = 0

        def read(self, size=-1):
            self.calls += 1
            if self.calls == 1:
                return super().read(8)
            if failure == "read_error":
                raise OSError("simulated input read failure")
            return "text is not binary"

    stream = FailingStream(png_bytes)
    expected = OSError if failure == "read_error" else CollectionError
    with pytest.raises(expected) as rejected:
        ImageCollectionService(settings).collect("读取失败", [ImageInput("front.png", stream)])
    if failure == "text_chunk":
        assert rejected.value.code == "invalid_images"
        assert rejected.value.status_code == 422
    assert not stream.closed
    assert not list(settings.data_dir.rglob("profile.json"))
    assert not list(settings.data_dir.rglob("*.upload"))
    assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("end_marker", [None, "", bytearray()])
def test_direct_collection_rejects_non_binary_end_marker(settings, png_bytes, end_marker):
    class NonBinaryEndStream(BytesIO):
        def read(self, size=-1):
            chunk = super().read(size)
            return chunk if chunk else end_marker

    stream = NonBinaryEndStream(png_bytes)
    with pytest.raises(CollectionError) as rejected:
        ImageCollectionService(settings).collect("无效结束标记", [ImageInput("front.png", stream)])
    assert rejected.value.code == "invalid_images"
    assert not stream.closed
    assert not list(settings.data_dir.rglob("profile.json"))
    assert list((settings.data_dir / ".staging").iterdir()) == []


def test_direct_collection_reads_from_current_stream_position_and_preserves_checksum(settings, png_bytes):
    import hashlib

    prefix = b"caller-owned prefix"
    stream = BytesIO(prefix + png_bytes)
    stream.seek(len(prefix))
    profile = ImageCollectionService(settings).collect("当前位置", [ImageInput("front.png", stream)])
    asset = profile.images[0]
    assert asset.size_bytes == len(png_bytes)
    assert asset.checksum == hashlib.sha256(png_bytes).hexdigest()
    assert (settings.data_dir / asset.storage_path).read_bytes() == png_bytes
    assert not stream.closed
