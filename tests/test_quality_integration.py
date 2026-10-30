"""Quality feedback must reach the saved archive without changing its contract."""

from copy import deepcopy
from io import BytesIO
import json
from uuid import uuid4

from PIL import Image, ImageDraw
import pytest

from app.main import create_app
from app.models.image_quality import ImageQualityData
from app.models.processing_result import ProcessingResult
from app.modules import image_metadata
from app.modules.image_metadata import ImageMetadataProcessor
from app.modules.image_quality import ImageQualityProcessor
from app.services.image_collection import ImageCollectionService, ImageInput
from app.services.pipeline import Pipeline
from app.services.profile_assembler import ProfileAssembler


def detailed_image(image_format="PNG"):
    image = Image.new("RGB", (960, 720), (64, 64, 64))
    draw = ImageDraw.Draw(image)
    for y in range(0, 720, 8):
        for x in range(0, 960, 8):
            if (x // 8 + y // 8) % 2:
                draw.rectangle((x, y, x + 7, y + 7), fill=(192, 192, 192))
    stream = BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


def dark_image():
    stream = BytesIO()
    Image.new("RGB", (960, 720), (0, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


def quality_result(profile):
    return next(result for result in profile["processing_results"] if result["module_name"] == "image_quality")


@pytest.mark.parametrize("image_format,mime", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"),
])
def test_default_upload_runs_real_quality_and_round_trips(client, settings, image_format, mime):
    content = detailed_image(image_format)
    response = client.post(
        "/api/v1/objects", data={"name": "清晰纹理示例"},
        files=[("images", ("same-name.photo", content, mime))],
    )
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert [result["module_name"] for result in profile["processing_results"]] == [
        "image_metadata", "image_quality", "object_segmentation",
    ]
    result = quality_result(profile)
    data = ImageQualityData.model_validate(result["data"])
    assert result["status"] == "success"
    assert (data.input_count, data.checked_count, data.review_count) == (1, 1, 0)
    checked = data.images[0]
    assert checked.status == "good"
    assert str(checked.asset_id) == profile["images"][0]["asset_id"]
    assert checked.filename == profile["images"][0]["filename"]
    assert (checked.width, checked.height) == (960, 720)
    assert 100 < checked.mean_luminance < 155
    assert checked.contrast_stddev > 40
    assert checked.edge_strength > 4
    assert profile["status"] == "completed_with_warnings"
    assert profile["processing_results"][-1]["status"] == "skipped"
    assert profile["processing_results"][-1]["data"]["images"][0]["status"] == "disabled"
    assert profile["schema_version"] == "0.1.0"
    assert profile["measurements"][0]["value"] is None
    assert profile["geometry"]["scale_status"] == "uncalibrated"

    object_id = profile["object_id"]
    stored = json.loads((settings.data_dir / "objects" / object_id / "profile.json").read_text(encoding="utf-8"))
    assert client.get(f"/api/v1/objects/{object_id}").json() == profile
    assert client.get(f"/api/v1/objects/{object_id}/profile.json").json() == stored == profile
    assert (settings.data_dir / profile["images"][0]["storage_path"]).read_bytes() == content


def test_quality_review_is_saved_with_suggestions_and_distinct_asset_ids(client, settings):
    response = client.post(
        "/api/v1/objects", data={"name": "重复文件名的不同照片"},
        files=[("images", ("same.png", detailed_image(), "image/png")),
               ("images", ("same.png", dark_image(), "image/png"))],
    )
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    result = quality_result(profile)
    data = ImageQualityData.model_validate(result["data"])
    assert result["status"] == "success"  # Quality review is a completed analysis, not an exception.
    assert (data.input_count, data.checked_count, data.review_count) == (2, 2, 1)
    by_id = {str(record.asset_id): record for record in data.images}
    assert by_id[profile["images"][0]["asset_id"]].status == "good"
    dark = by_id[profile["images"][1]["asset_id"]]
    assert dark.status == "review"
    assert "too_dark" in dark.issues
    assert dark.suggestions
    assert result["warnings"] and profile["warnings"]
    assert profile["status"] == "completed_with_warnings"
    assert len(list((settings.data_dir / "objects").iterdir())) == 1
    assert list((settings.data_dir / ".staging").iterdir()) == []


def test_default_collection_reuses_validation_and_custom_processors_stay_supported(settings, monkeypatch):
    calls = []
    original = image_metadata.inspect_image

    def inspect(path, max_pixels=40_000_000):
        calls.append(path)
        return original(path, max_pixels)

    monkeypatch.setattr(image_metadata, "inspect_image", inspect)
    service = ImageCollectionService(settings)
    profile = service.collect("直接采集", [ImageInput("texture.png", BytesIO(detailed_image()))])
    assert len(calls) == 1
    assert [result.module_name for result in profile.processing_results] == [
        "image_metadata", "image_quality", "object_segmentation",
    ]
    assert service.get_profile(profile.object_id) == profile

    custom = ImageCollectionService(settings, processors=[ImageMetadataProcessor()])
    custom_profile = custom.collect("自定义模块", [ImageInput("texture.png", BytesIO(detailed_image()))])
    assert [result.module_name for result in custom_profile.processing_results] == ["image_metadata"]
    supplied = Pipeline([ImageQualityProcessor()])
    assert ImageCollectionService(settings, pipeline=supplied).pipeline is supplied
    assert create_app(settings, pipeline=supplied).state.collection_service.pipeline is supplied
    with pytest.raises(ValueError, match="either"):
        ImageCollectionService(settings, processors=[], pipeline=supplied)


def test_existing_archive_without_quality_remains_readable_without_being_rewritten(client, settings):
    legacy_service = ImageCollectionService(settings, processors=[ImageMetadataProcessor()])
    existing = legacy_service.collect("已有基础档案", [ImageInput("texture.png", BytesIO(detailed_image()))])
    path = settings.data_dir / "objects" / str(existing.object_id) / "profile.json"
    original_bytes = path.read_bytes()
    expected = existing.model_dump(mode="json")
    assert client.get(f"/api/v1/objects/{existing.object_id}").json() == expected
    assert client.get(f"/api/v1/objects/{existing.object_id}/profile.json").content == original_bytes
    assert [result.module_name for result in existing.processing_results] == ["image_metadata"]
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("all_fail", [False, True])
def test_quality_errors_preserve_other_images_and_the_downloadable_archive(client, settings, monkeypatch, all_fail):
    original = ImageQualityProcessor._assess_asset

    def assess(self, context, asset):
        if all_fail or asset.filename == "unavailable.png":
            raise OSError("PRIVATE_QUALITY_PATH_C:\\private\\analysis")
        return original(self, context, asset)

    monkeypatch.setattr(ImageQualityProcessor, "_assess_asset", assess)
    response = client.post(
        "/api/v1/objects", data={"name": "保留可用评估"},
        files=[("images", ("ok.png", detailed_image(), "image/png")),
               ("images", ("unavailable.png", detailed_image(), "image/png"))],
    )
    assert response.status_code == 201, response.text
    assert "PRIVATE_QUALITY_PATH" not in response.text
    profile = response.json()["profile"]
    result = quality_result(profile)
    data = ImageQualityData.model_validate(result["data"])
    assert result["status"] == ("failed" if all_fail else "partial")
    assert data.input_count == 2
    assert data.checked_count == (0 if all_fail else 1)
    assert data.review_count == 0
    assert len(result["errors"]) == (2 if all_fail else 1)
    assert all(error["asset_id"] and error["error_id"] for error in result["errors"])
    assert profile["status"] == "completed_with_warnings"
    assert len(profile["images"]) == 2
    if not all_fail:
        assert str(data.images[0].asset_id) == profile["images"][0]["asset_id"]
    assert client.get(f"/api/v1/objects/{profile['object_id']}/profile.json").json() == profile
    assert all((settings.data_dir / image["storage_path"]).is_file() for image in profile["images"])
    assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("fault", ["foreign_asset", "wrong_filename", "wrong_count", "missing_success_image"])
def test_quality_adapter_rejects_mismatched_results_without_changing_profile(settings, fault):
    source = ImageCollectionService(settings).collect("不可覆盖身份", [ImageInput("texture.png", BytesIO(detailed_image()))])
    data = deepcopy(source.processing_results[1].data)
    if fault == "foreign_asset":
        data["images"][0]["asset_id"] = str(uuid4())
    elif fault == "wrong_filename":
        data["images"][0]["filename"] = "other.png"
    elif fault == "wrong_count":
        data["input_count"] = 2
    else:
        data.update(images=[], checked_count=0, review_count=0)
    result = ProcessingResult(module_name="image_quality", status="success", data=data)
    assembler = ProfileAssembler()
    original = assembler.create_profile(source.name, source.images, source.object_id)
    merged = assembler.merge_result(original, result)
    assert merged.processing_results[-1].status == "failed"
    assert merged.processing_results[-1].errors[-1].code == "invalid_module_data"
    assert merged.object_id == original.object_id
    assert merged.created_at == original.created_at
    assert merged.images == original.images
    assert merged.name == original.name
    assert original.processing_results == []
