import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.object_profile import ObjectProfile
from app.storage.local_storage import ObjectTransaction
from conftest import image_bytes


def upload(client, content, name="测试物体", filename="front.png", mime="image/png"):
    return client.post(
        "/api/v1/objects",
        data={"name": name},
        files=[("images", (filename, content, mime))],
    )


def assert_safe_error(response, expected_status):
    assert response.status_code == expected_status, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"]
    assert detail["message"]
    assert "Traceback" not in response.text


def test_health_and_chinese_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "物体" in page.text
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize(
    ("image_format", "filename", "mime"),
    [("PNG", "front.png", "image/png"), ("JPEG", "front.jpg", "image/jpeg"), ("WEBP", "front.webp", "image/webp")],
)
def test_upload_supported_formats_and_persist(client, settings, image_format, filename, mime):
    content = image_bytes(image_format)
    response = upload(client, content, filename=filename, mime=mime)
    assert response.status_code == 201, response.text
    created = response.json()
    profile = created["profile"]
    assert str(UUID(created["object_id"])) == profile["object_id"]
    assert created["status"] == profile["status"]
    asset = profile["images"][0]
    assert (asset["width"], asset["height"]) == (32, 24)
    assert asset["mime_type"] == mime
    assert asset["size_bytes"] == len(content)
    relative = Path(asset["storage_path"])
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    assert (settings.data_dir / relative).read_bytes() == content
    result = next(item for item in profile["processing_results"] if item["module_name"] == "image_metadata")
    assert result["status"] == "success"
    metadata = result["data"]["images"][0]
    assert metadata["asset_id"] == asset["asset_id"]
    assert metadata["image_format"] == image_format
    assert (metadata["width"], metadata["height"], metadata["size_bytes"]) == (32, 24, len(content))


def test_multiple_images_link_to_one_object_with_distinct_ids(client, settings, png_bytes):
    response = client.post(
        "/api/v1/objects",
        data={"name": "相同物体的不同角度"},
        files=[("images", ("same.png", png_bytes, "image/png")), ("images", ("same.png", png_bytes, "image/png"))],
    )
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert len(profile["images"]) == 2
    assert len({asset["asset_id"] for asset in profile["images"]}) == 2
    assert all(asset["storage_path"].replace("\\", "/").startswith(f"objects/{profile['object_id']}/images/") for asset in profile["images"])
    assert len(list((settings.data_dir / "objects").iterdir())) == 1


def test_created_fetched_downloaded_and_saved_json_are_identical(client, settings, png_bytes):
    created = upload(client, png_bytes).json()
    object_id = created["object_id"]
    fetched = client.get(f"/api/v1/objects/{object_id}")
    downloaded = client.get(f"/api/v1/objects/{object_id}/profile.json")
    assert fetched.status_code == downloaded.status_code == 200
    assert "attachment" in downloaded.headers["content-disposition"]
    persisted = (settings.data_dir / "objects" / object_id / "profile.json").read_bytes()
    assert created["profile"] == fetched.json() == downloaded.json() == json.loads(persisted)
    assert ObjectProfile.model_validate_json(persisted).object_id == UUID(object_id)


def test_persisted_profile_survives_new_app_instance(client, settings, png_bytes):
    profile = upload(client, png_bytes).json()["profile"]
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/api/v1/objects/{profile['object_id']}").json() == profile


def test_actual_image_content_controls_type(client, png_bytes):
    response = upload(client, png_bytes, filename="misleading.txt", mime="text/plain")
    assert response.status_code == 201, response.text
    assert response.json()["profile"]["images"][0]["mime_type"] == "image/png"


def test_non_image_is_rejected(client):
    response = upload(client, b"This is text, not an image.", filename="note.txt", mime="text/plain")
    assert response.status_code in (415, 422)
    assert isinstance(response.json()["detail"], dict)


def test_decodable_but_unsupported_format_is_rejected(client):
    assert_safe_error(upload(client, image_bytes("BMP"), filename="image.bmp", mime="image/bmp"), 415)


def test_corrupt_image_is_rejected(client, png_bytes):
    assert_safe_error(upload(client, png_bytes[:40]), 422)


def test_truncated_pixel_data_is_rejected_by_actual_decoding(client):
    content = image_bytes("JPEG", size=(100, 100))
    assert_safe_error(upload(client, content[:-20], filename="truncated.jpg", mime="image/jpeg"), 422)


def test_invalid_batch_leaves_no_uploaded_files(client, settings, png_bytes):
    response = client.post(
        "/api/v1/objects",
        data={"name": "应全部撤销"},
        files=[("images", ("good.png", png_bytes, "image/png")), ("images", ("broken.png", png_bytes[:40], "image/png"))],
    )
    assert_safe_error(response, 422)
    assert not list(settings.data_dir.rglob("*.png"))
    assert not list(settings.data_dir.rglob("profile.json"))
    objects = settings.data_dir / "objects"
    assert not objects.exists() or not list(objects.iterdir())


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_rejected(client, png_bytes, name):
    assert_safe_error(upload(client, png_bytes, name=name), 422)


def test_missing_images_rejected(client):
    assert_safe_error(client.post("/api/v1/objects", data={"name": "没有图片"}), 422)


def test_single_image_size_limit(client_factory, png_bytes):
    client, _ = client_factory(max_image_bytes=len(png_bytes) - 1)
    assert_safe_error(upload(client, png_bytes), 413)


def test_batch_size_limit(client_factory, png_bytes):
    client, _ = client_factory(max_total_bytes=len(png_bytes) * 2 - 1)
    response = client.post(
        "/api/v1/objects", data={"name": "超总量"},
        files=[("images", ("one.png", png_bytes, "image/png")), ("images", ("two.png", png_bytes, "image/png"))],
    )
    assert_safe_error(response, 413)


def test_file_count_limit(client_factory, png_bytes):
    client, _ = client_factory(max_files=1)
    response = client.post(
        "/api/v1/objects", data={"name": "超数量"},
        files=[("images", ("one.png", png_bytes, "image/png")), ("images", ("two.png", png_bytes, "image/png"))],
    )
    assert_safe_error(response, 422)


def test_decoded_pixel_limit(client_factory, png_bytes):
    client, _ = client_factory(max_image_pixels=32 * 24 - 1)
    assert_safe_error(upload(client, png_bytes), 413)


def test_client_filename_cannot_control_storage_path(client, settings, png_bytes):
    response = upload(client, png_bytes, filename="../../outside.png")
    assert response.status_code == 201, response.text
    asset = response.json()["profile"]["images"][0]
    actual_path = (settings.data_dir / asset["storage_path"]).resolve()
    assert actual_path.is_relative_to(settings.data_dir.resolve())
    assert actual_path.stem == asset["asset_id"]
    assert not (settings.data_dir.parent / "outside.png").exists()
    assert str(settings.data_dir.resolve()) not in response.text


def test_unknown_object_returns_clear_not_found(client):
    object_id = uuid4()
    assert_safe_error(client.get(f"/api/v1/objects/{object_id}"), 404)
    assert_safe_error(client.get(f"/api/v1/objects/{object_id}/profile.json"), 404)


def test_storage_failure_is_safe_and_rolls_back(client, settings, png_bytes, monkeypatch):
    secret = "INTERNAL_STORAGE_ERROR_C:\\private\\storage"

    def fail_to_save(self, profile):
        raise OSError(secret)

    monkeypatch.setattr(ObjectTransaction, "save_profile", fail_to_save)
    with TestClient(create_app(settings), raise_server_exceptions=False) as error_client:
        response = upload(error_client, png_bytes)
    assert_safe_error(response, 500)
    assert secret not in response.text
    assert not list(settings.data_dir.rglob("profile.json"))
    assert not list(settings.data_dir.rglob("*.png"))
