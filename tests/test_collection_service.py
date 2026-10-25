"""The parent project can collect images without invoking the HTTP adapter."""

from io import BytesIO
import json

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
