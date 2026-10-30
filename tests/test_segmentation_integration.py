"""HTTP and archive regressions; model inference is injected only in tests."""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from app.config import Settings
from app.main import create_app
from app.models.segmentation import NormalizedBox, SegmentationData
from app.modules.image_metadata import ImageMetadataProcessor
from app.modules.image_quality import ImageQualityProcessor
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.modules.segmentation_backend import MaskCandidate, MaskPrediction, SegmentationCapabilities
from app.services.image_collection import CollectionError, ImageCollectionService, ImageInput
from app.storage import local_storage
from app.storage.local_storage import ObjectTransaction, storage_io_path


class TestOnlyBackend:
    __test__ = False
    model_id = "test_only"
    model_name = "baidu_intelligent_segmentation"
    model_version = "test-fixture"
    capabilities = SegmentationCapabilities(box_prompt=True, multiple_masks=True, quality_score=True)

    def __init__(self, fail_second=False):
        self.calls = []
        self.fail_second = fail_second

    def predict(self, request):
        image, box = request.image, request.prompt.box
        self.calls.append((image.size, box))
        if self.fail_second and len(self.calls) == 2:
            raise RuntimeError("PRIVATE_SEGMENTATION_PATH_C:\\private\\model")
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).ellipse((image.width // 4, image.height // 4,
                                     image.width * 3 // 4, image.height * 3 // 4), fill=255)
        return MaskPrediction(mask, 0.9, score_type="predicted_iou")


def configured_client(settings, backend):
    settings = replace(settings, segmentation_enabled=True)
    processors = [ImageMetadataProcessor(), ImageQualityProcessor(), ObjectSegmentationProcessor(settings, backend=backend)]
    return TestClient(create_app(settings, processors=processors)), settings


def post(client, content, *, boxes=None, count=1):
    data = {"name": "框选前景"}
    if boxes is not None:
        data["segmentation_boxes"] = boxes if isinstance(boxes, str) else json.dumps(boxes)
    return client.post("/api/v1/objects", data=data,
                       files=[("images", ("same.png", content, "image/png")) for _ in range(count)])


def result_of(profile):
    return next(item for item in profile["processing_results"] if item["module_name"] == "object_segmentation")


def test_default_disabled_keeps_upload_quality_and_existing_archive_shape(client, settings, png_bytes):
    response = post(client, png_bytes)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert [item["module_name"] for item in profile["processing_results"]] == [
        "image_metadata", "image_quality", "object_segmentation",
    ]
    result = result_of(profile)
    assert result["status"] == "skipped"
    record = SegmentationData.model_validate(result["data"]).images[0]
    assert record.status == "disabled"
    assert record.model_name == "baidu_intelligent_segmentation"
    assert record.mask_path is record.bbox is record.area_ratio is None
    assert record.warnings
    assert profile["schema_version"] == "0.1.0"
    assert "segmentation" not in profile
    assert client.get(f"/api/v1/objects/{profile['object_id']}").json() == profile
    assert not list(settings.data_dir.rglob("masks/*.png"))
    config = client.get("/api/v1/config").json()
    assert config["segmentation_enabled"] is False
    assert config["segmentation_prompt_type"] == "box"
    assert config["segmentation_model"] == "baidu_intelligent_segmentation"
    assert "checkpoint" not in json.dumps(config)


def test_http_box_produces_owned_png_and_safe_previews(settings, png_bytes):
    backend = TestOnlyBackend()
    client, settings = configured_client(settings, backend)
    with client:
        response = post(client, png_bytes, boxes=[[0.1, 0.2, 0.9, 0.8]])
        assert response.status_code == 201, response.text
        profile = response.json()["profile"]
        result = result_of(profile)
        data = SegmentationData.model_validate(result["data"])
        assert (data.success_count, data.review_count, data.failed_count, data.skipped_count) == (1, 0, 0, 0)
        record = data.images[0]
        asset = profile["images"][0]
        assert str(record.asset_id) == asset["asset_id"]
        assert record.mask_path.startswith(f"objects/{profile['object_id']}/masks/{asset['asset_id']}_")
        assert record.coordinate_system == "exif_oriented_pixels"
        assert backend.calls[0][0] == (32, 24)
        assert backend.calls[0][1] == pytest.approx((3.2, 4.8, 28.8, 19.2))
        mask_bytes = storage_io_path(settings.data_dir / record.mask_path).read_bytes()
        with Image.open(BytesIO(mask_bytes)) as mask:
            assert mask.format == "PNG" and mask.mode == "L" and mask.size == (32, 24)
            assert {value for _, value in mask.getcolors()} == {0, 255}
            assert mask.getbbox() == record.bbox
        base = f"/api/v1/objects/{profile['object_id']}/images/{asset['asset_id']}"
        original, mask_preview = client.get(base + "/original"), client.get(base + "/mask")
        assert original.status_code == mask_preview.status_code == 200
        assert original.content == png_bytes
        assert mask_preview.content == mask_bytes
        assert mask_preview.headers["content-type"] == "image/png"
        assert mask_preview.headers["x-content-type-options"] == "nosniff"
        assert client.get(f"/api/v1/objects/{profile['object_id']}/profile.json").json() == profile
        assert client.get(f"/api/v1/objects/{profile['object_id']}").json() == profile
        assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("boxes", ["not-json", "{}", "[]", "[null,null]", "[[0,0,0,1]]",
                                   "[[1,0,0,1]]", "[[0,0,2,1]]", "[[0,0,NaN,1]]",
                                   '[["0",0,1,1]]', "[[false,0,1,1]]"])
def test_invalid_prompt_json_is_rejected_before_publishing(client, settings, png_bytes, boxes):
    response = post(client, png_bytes, boxes=boxes)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_segmentation_boxes"
    assert not list(settings.data_dir.rglob("profile.json"))
    assert not list((settings.data_dir / ".multipart").iterdir())


def test_direct_collection_revalidates_box_even_without_http(settings, png_bytes):
    prompt = NormalizedBox(x0=0, y0=0, x1=1, y1=1).model_copy(update={"x1": 2})
    stream = BytesIO(png_bytes)
    with pytest.raises(CollectionError) as failure:
        ImageCollectionService(settings).collect("无效目标框", [ImageInput("photo.png", stream, prompt)])
    assert failure.value.code == "invalid_segmentation_boxes"
    assert stream.tell() == 0 and not stream.closed
    assert not settings.data_dir.exists()


def test_missing_box_and_missing_credentials_are_explicit_not_fake_success(settings, png_bytes):
    enabled = replace(settings, segmentation_enabled=True, segmentation_api_key="", segmentation_secret_key="")
    with TestClient(create_app(enabled)) as client:
        missing = post(client, png_bytes).json()["profile"]
        assert result_of(missing)["data"]["images"][0]["status"] == "needs_input"
        unavailable = post(client, png_bytes, boxes=[[0, 0, 1, 1]]).json()["profile"]
        record = result_of(unavailable)["data"]["images"][0]
        assert record["status"] == "unavailable"
        assert record["error_id"] and record["mask_path"] is None
        assert "API Key" in "".join(record["warnings"]) and "Secret Key" in "".join(record["warnings"])
        asset = unavailable["images"][0]
        assert client.get(f"/api/v1/objects/{unavailable['object_id']}/images/{asset['asset_id']}/mask").status_code == 404


def test_multiview_failure_retains_other_mask_and_originals(settings, png_bytes):
    backend = TestOnlyBackend(fail_second=True)
    client, settings = configured_client(settings, backend)
    with client:
        response = post(client, png_bytes, count=2, boxes=[[0, 0, 1, 1], [0, 0, 1, 1]])
        assert response.status_code == 201, response.text
        assert "PRIVATE_SEGMENTATION_PATH" not in response.text
        profile = response.json()["profile"]
        data = SegmentationData.model_validate(result_of(profile)["data"])
        assert result_of(profile)["status"] == "partial"
        assert (data.success_count, data.failed_count) == (1, 1)
        assert [str(image.asset_id) for image in data.images] == [image["asset_id"] for image in profile["images"]]
        assert len(list((settings.data_dir / "objects" / profile["object_id"] / "masks").glob("*.png"))) == 1
        assert all((settings.data_dir / image["storage_path"]).read_bytes() == png_bytes for image in profile["images"])


@pytest.mark.parametrize("phase", ["profile", "commit"])
def test_mask_rolls_back_with_archive_when_publication_fails(settings, png_bytes, monkeypatch, phase):
    client, settings = configured_client(settings, TestOnlyBackend())
    original = local_storage.os.replace
    observed = []

    def fail(source, destination):
        source, destination = Path(source), Path(destination)
        if ((phase == "profile" and destination.name == "profile.json")
                or (phase == "commit" and destination.parent == settings.data_dir / "objects")):
            parent = source.parent if phase == "profile" else source
            assert list((parent / "masks").glob("*.png"))
            observed.append(True)
            raise OSError("injected publication failure")
        return original(source, destination)

    monkeypatch.setattr(local_storage.os, "replace", fail)
    with TestClient(client.app, raise_server_exceptions=False) as tolerant:
        assert post(tolerant, png_bytes, boxes=[[0, 0, 1, 1]]).status_code == 500
    assert observed
    assert list((settings.data_dir / ".staging").iterdir()) == []
    assert not list(settings.data_dir.rglob("profile.json"))
    assert not list(settings.data_dir.rglob("*.png"))


def test_mask_write_failure_is_isolated_without_discarding_valid_input(settings, png_bytes, monkeypatch):
    def fail(*args):
        raise OSError("PRIVATE_ARTIFACT_PATH")

    monkeypatch.setattr(ObjectTransaction, "save_mask", fail)
    client, settings = configured_client(settings, TestOnlyBackend())
    with client:
        response = post(client, png_bytes, boxes=[[0, 0, 1, 1]])
        assert response.status_code == 201, response.text
        assert "PRIVATE_ARTIFACT_PATH" not in response.text
        profile = response.json()["profile"]
        assert result_of(profile)["data"]["images"][0]["status"] == "failed"
        assert (settings.data_dir / profile["images"][0]["storage_path"]).read_bytes() == png_bytes


@pytest.mark.parametrize("tamper", ["foreign_path", "missing_mask", "wrong_bbox", "wrong_area", "latest_failed"])
def test_preview_rejects_unowned_missing_or_inconsistent_masks(settings, png_bytes, tamper):
    client, settings = configured_client(settings, TestOnlyBackend())
    with client:
        profile = post(client, png_bytes, boxes=[[0, 0, 1, 1]]).json()["profile"]
        result = result_of(profile)
        record = result["data"]["images"][0]
        if tamper == "foreign_path":
            record["mask_path"] = record["mask_path"].replace(profile["object_id"], str(uuid4()))
        elif tamper == "missing_mask":
            storage_io_path(settings.data_dir / record["mask_path"]).unlink()
        elif tamper == "wrong_bbox":
            record["bbox"] = [0, 0, 1, 1]
        elif tamper == "wrong_area":
            record["area_ratio"] = 0.001
        else:
            profile["processing_results"].append({"module_name": "object_segmentation", "status": "failed",
                                                   "data": {}, "warnings": [], "errors": []})
        path = settings.data_dir / "objects" / profile["object_id"] / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        base = f"/api/v1/objects/{profile['object_id']}/images/{record['asset_id']}"
        assert client.get(base + "/mask").status_code == 404
        assert client.get(f"/api/v1/objects/{profile['object_id']}/images/{uuid4()}/original").status_code == 404


def test_exif_orientation_keeps_prompt_mask_and_preview_in_one_coordinate_system(settings):
    stream = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (32, 24), "gray").save(stream, format="JPEG", exif=exif)
    content = stream.getvalue()
    backend = TestOnlyBackend()
    client, settings = configured_client(settings, backend)
    with client:
        profile = post(client, content, boxes=[[0.25, 0, 0.75, 1]]).json()["profile"]
        record = SegmentationData.model_validate(result_of(profile)["data"]).images[0]
        assert (record.width, record.height) == (24, 32)
        assert backend.calls[0] == ((24, 32), (6, 0, 18, 32))
        assert (profile["images"][0]["width"], profile["images"][0]["height"]) == (32, 24)
        assert (settings.data_dir / profile["images"][0]["storage_path"]).read_bytes() == content


def test_long_data_root_commits_and_previews_masks_without_leaking_windows_prefix(tmp_path, png_bytes):
    # The original and final mask paths exceed MAX_PATH, independently of pytest's base directory.
    padding = max(16, 190 - len(str(tmp_path)) - len("/long-data-"))
    settings = Settings(data_dir=tmp_path / ("long-data-" + "x" * padding))
    client, settings = configured_client(settings, TestOnlyBackend())
    with client:
        response = post(client, png_bytes, boxes=[[0, 0, 1, 1]])
        assert response.status_code == 201, response.text
        profile = response.json()["profile"]
        record = SegmentationData.model_validate(result_of(profile)["data"]).images[0]
        assert record.status == "success", result_of(profile)
        assert len(str(settings.data_dir / record.mask_path)) > 260
        assert "\\" not in record.mask_path and not record.mask_path.startswith("/")
        base = f"/api/v1/objects/{profile['object_id']}/images/{record.asset_id}"
        original = client.get(base + "/original")
        mask = client.get(base + "/mask")
        assert original.status_code == mask.status_code == 200
        assert original.content == png_bytes
        assert mask.content == storage_io_path(settings.data_dir / record.mask_path).read_bytes()
        assert client.get(f"/api/v1/objects/{profile['object_id']}/profile.json").json() == profile
        assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("fault", ["object", "asset"])
def test_original_preview_rejects_cross_resource_paths_even_when_the_file_exists(client, settings, png_bytes, fault):
    profile = post(client, png_bytes).json()["profile"]
    asset = profile["images"][0]
    wrong_path = asset["storage_path"].replace(
        profile["object_id"] if fault == "object" else asset["asset_id"], str(uuid4())
    )
    target = settings.data_dir / wrong_path
    storage_io_path(target.parent).mkdir(parents=True, exist_ok=True)
    storage_io_path(target).write_bytes(png_bytes)
    asset["storage_path"] = wrong_path
    archive = settings.data_dir / "objects" / profile["object_id"] / "profile.json"
    storage_io_path(archive).write_text(json.dumps(profile), encoding="utf-8")
    response = client.get(f"/api/v1/objects/{profile['object_id']}/images/{asset['asset_id']}/original")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "image_not_found"


@pytest.mark.parametrize("name,value", [
    ("segmentation_enabled", 1), ("segmentation_timeout_seconds", 0),
    ("segmentation_timeout_seconds", 121), ("segmentation_timeout_seconds", float("nan")),
    ("segmentation_timeout_seconds", True), ("segmentation_api_key", "bad\nkey"),
    ("segmentation_secret_key", 123),
])
def test_segmentation_configuration_rejects_ambiguous_values(tmp_path, name, value):
    with pytest.raises(ValueError):
        Settings(data_dir=tmp_path, **{name: value})


def test_api_credentials_are_absent_from_repr_public_config_archive_and_logs(settings, png_bytes, caplog):
    api_key, secret_key = "UNIT_API_KEY_DO_NOT_EXPOSE", "UNIT_SECRET_DO_NOT_EXPOSE"
    settings = replace(settings, segmentation_enabled=False,
                       segmentation_api_key=api_key, segmentation_secret_key=secret_key)
    assert api_key not in repr(settings) and secret_key not in repr(settings)
    with TestClient(create_app(settings)) as client:
        config = client.get("/api/v1/config")
        created = post(client, png_bytes)
        assert created.status_code == 201, created.text
        profile = created.json()["profile"]
        archive = storage_io_path(settings.data_dir / "objects" / profile["object_id"] / "profile.json").read_text(encoding="utf-8")
        for output in (config.text, created.text, archive, caplog.text):
            assert api_key not in output and secret_key not in output
        assert "api_key" not in config.text and "secret_key" not in config.text


def test_legacy_sam2_archive_remains_previewable_while_new_records_identify_cloud_backend(settings, png_bytes):
    client, settings = configured_client(settings, TestOnlyBackend())
    with client:
        profile = post(client, png_bytes, boxes=[[0, 0, 1, 1]]).json()["profile"]
        current_record = result_of(profile)["data"]["images"][0]
        assert current_record["model_name"] == "baidu_intelligent_segmentation"
        current_record["model_name"] = "sam2.1_hiera_tiny"
        archive = storage_io_path(settings.data_dir / "objects" / profile["object_id"] / "profile.json")
        archive.write_text(json.dumps(profile), encoding="utf-8")
        old_bytes = archive.read_bytes()
        fetched = client.get(f"/api/v1/objects/{profile['object_id']}")
        assert fetched.status_code == 200
        assert result_of(fetched.json())["data"]["images"][0]["model_name"] == "sam2.1_hiera_tiny"
        base = f"/api/v1/objects/{profile['object_id']}/images/{current_record['asset_id']}"
        preview = client.get(base + "/mask")
        assert preview.status_code == 200
        assert preview.content == storage_io_path(settings.data_dir / current_record["mask_path"]).read_bytes()
        assert archive.read_bytes() == old_bytes
        new_profile = post(client, png_bytes, boxes=[[0, 0, 1, 1]]).json()["profile"]
        assert result_of(new_profile)["data"]["images"][0]["model_name"] == "baidu_intelligent_segmentation"
