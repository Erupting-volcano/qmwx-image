"""Uploading a sweep instead of photos: service and HTTP behaviour."""

from io import BytesIO

from PIL import Image

from app.services.image_collection import ImageCollectionService
from video_helpers import make_video


def test_collect_accepts_a_video_sweep(settings):
    profile = ImageCollectionService(settings).collect(
        "视频物体", [], video=make_video(2.0), video_filename="sweep.mp4")
    assert len(profile.images) == 3
    assert all(image.filename.startswith("frame-") for image in profile.images)
    for image in profile.images:
        assert image.width == 160 and image.height == 120


def test_upload_with_video_creates_an_archive(client):
    payload = make_video(2.0).getvalue()
    response = client.post("/api/v1/objects", data={"name": "视频物体"},
                           files={"video": ("sweep.mp4", payload, "video/mp4")})
    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["profile"]["images"]) == 3
    assert body["profile"]["images"][0]["filename"].startswith("frame-")


def test_images_and_video_cannot_be_combined(client, png_bytes):
    response = client.post("/api/v1/objects", data={"name": "混用"},
                           files={"images": ("a.png", png_bytes, "image/png"),
                                  "video": ("sweep.mp4", make_video(1.0).getvalue(), "video/mp4")})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_input"


def test_broken_video_returns_a_clear_error(client):
    response = client.post("/api/v1/objects", data={"name": "坏视频"},
                           files={"video": ("bad.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_video"


def test_extracted_frames_are_real_jpegs(client):
    payload = make_video(2.0).getvalue()
    created = client.post("/api/v1/objects", data={"name": "帧检查"},
                          files={"video": ("sweep.mp4", payload, "video/mp4")}).json()
    object_id = created["object_id"]
    asset = created["profile"]["images"][0]
    preview = client.get(f"/api/v1/objects/{object_id}/images/{asset['asset_id']}/original")
    assert preview.status_code == 200
    image = Image.open(BytesIO(preview.content))
    assert image.format == "JPEG"
