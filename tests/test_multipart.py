"""Multipart integrity and temporary storage boundaries of the HTTP adapter."""

import asyncio
from io import BytesIO
import json
from pathlib import Path
import tempfile

from PIL import Image
import pytest

from app.api import routes
from app.config import Settings
from app.main import create_app


BOUNDARY = b"multipart-regression-boundary"
CONTENT_TYPE = "multipart/form-data; boundary=" + BOUNDARY.decode("ascii")


@pytest.fixture
def large_png():
    buffer = BytesIO()
    Image.new("RGB", (640, 640), (30, 110, 170)).save(buffer, format="PNG", compress_level=0)
    content = buffer.getvalue()
    assert len(content) > 1024 * 1024
    return content


@pytest.fixture
def upload_temporary_files(monkeypatch):
    spools = []
    disk_files = []
    original_spool = routes.SpooledTemporaryFile
    original_disk_file = tempfile.TemporaryFile

    def record_spool(*args, **kwargs):
        spool = original_spool(*args, **kwargs)
        spools.append(spool)
        return spool

    def record_disk_file(*args, **kwargs):
        disk_file = original_disk_file(*args, **kwargs)
        disk_files.append((disk_file, Path(kwargs["dir"]).resolve()))
        return disk_file

    monkeypatch.setattr(routes, "SpooledTemporaryFile", record_spool)
    monkeypatch.setattr(tempfile, "TemporaryFile", record_disk_file)
    return spools, disk_files


def multipart_prefix():
    return (
        b"--" + BOUNDARY + b'\r\nContent-Disposition: form-data; name="name"\r\n\r\nTest object\r\n'
        + b"--" + BOUNDARY
        + b'\r\nContent-Disposition: form-data; name="images"; filename="image.png"\r\n'
        + b"Content-Type: image/png\r\n\r\n"
    )


def assert_no_collection(data_dir):
    assert not list(data_dir.rglob("profile.json"))
    assert not list(data_dir.rglob("*.png"))
    assert not list(data_dir.rglob("*.upload"))
    spool_dir = data_dir / ".multipart"
    assert not spool_dir.exists() or not list(spool_dir.iterdir())
    staging_dir = data_dir / ".staging"
    assert not staging_dir.exists() or not list(staging_dir.iterdir())


@pytest.mark.parametrize("second_image_complete", [False, True])
def test_truncated_multipart_rejects_entire_batch_and_closes_spools(
    client, settings, png_bytes, large_png, upload_temporary_files, second_image_complete,
):
    body = (
        multipart_prefix() + png_bytes + b"\r\n--" + BOUNDARY
        + b'\r\nContent-Disposition: form-data; name="images"; filename="unfinished.png"\r\n'
        + b"Content-Type: image/png\r\n\r\n"
        + (large_png if second_image_complete else large_png[:20])
    )
    # Both variants omit the closing boundary; the first completed image must
    # never turn a truncated batch into a successful one-image collection.
    response = client.post("/api/v1/objects", content=body, headers={"Content-Type": CONTENT_TYPE})

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_upload_form"
    spools, disk_files = upload_temporary_files
    assert len(spools) == 2
    assert all(spool.closed for spool in spools)
    if second_image_complete:
        assert disk_files
        assert all(file.closed for file, _ in disk_files)
    assert_no_collection(settings.data_dir)


def test_declared_request_body_over_limit_is_rejected_before_parsing(client_factory):
    client, settings = client_factory(max_total_bytes=1024)
    limit = settings.max_total_bytes + 1024 * 1024
    response = client.post(
        "/api/v1/objects", content=b"",
        headers={"Content-Type": CONTENT_TYPE, "Content-Length": str(limit + 1)},
    )

    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "upload_too_large"
    assert_no_collection(settings.data_dir)


def test_streamed_body_over_limit_closes_existing_spools(settings, upload_temporary_files):
    # Drive separate ASGI chunks so the first one actually rolls to disk before
    # the next exceeds the limit; no Content-Length header is supplied.
    settings = Settings(data_dir=settings.data_dir, max_total_bytes=1024 * 1024)
    application = create_app(settings)
    chunks = [multipart_prefix() + b"x" * (1024 * 1024 + 4096), b"x" * (1024 * 1024)]
    messages = []

    async def request():
        pending = iter(chunks)

        async def receive():
            chunk = next(pending, None)
            if chunk is None:
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": chunk, "more_body": True}

        async def send(message):
            messages.append(message)

        await application({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/api/v1/objects",
            "raw_path": b"/api/v1/objects", "query_string": b"",
            "headers": [(b"content-type", CONTENT_TYPE.encode("ascii"))],
            "client": ("127.0.0.1", 12345), "server": ("testserver", 80),
        }, receive, send)

    asyncio.run(request())

    start = next(message for message in messages if message["type"] == "http.response.start")
    payload = json.loads(b"".join(message.get("body", b"") for message in messages
                                  if message["type"] == "http.response.body"))
    assert start["status"] == 413
    assert payload["detail"]["code"] == "upload_too_large"
    spools, disk_files = upload_temporary_files
    assert len(spools) == 1
    assert spools[0]._rolled
    assert disk_files
    assert all(spool.closed for spool in spools)
    assert all(file.closed for file, _ in disk_files)
    assert_no_collection(settings.data_dir)


def test_large_upload_spools_inside_data_directory_and_cleans_up(
    client, settings, large_png, upload_temporary_files,
):
    response = client.post(
        "/api/v1/objects", data={"name": "Large image"},
        files=[("images", ("large.png", large_png, "image/png"))],
    )

    assert response.status_code == 201, response.text
    spools, disk_files = upload_temporary_files
    assert len(spools) == 1
    assert spools[0]._rolled
    assert disk_files
    assert all(directory == settings.data_dir / ".multipart" for _, directory in disk_files)
    assert all(spool.closed for spool in spools)
    assert all(file.closed for file, _ in disk_files)
    assert not list((settings.data_dir / ".multipart").iterdir())
    asset = response.json()["profile"]["images"][0]
    assert (settings.data_dir / asset["storage_path"]).read_bytes() == large_png


def test_rolled_spool_and_staging_are_cleaned_when_later_image_is_invalid(
    client, settings, large_png, upload_temporary_files,
):
    response = client.post(
        "/api/v1/objects", data={"name": "整批撤销"},
        files=[("images", ("large.png", large_png, "image/png")),
               ("images", ("broken.png", b"broken image", "image/png"))],
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_image"
    spools, disk_files = upload_temporary_files
    assert any(spool._rolled for spool in spools)
    assert all(spool.closed for spool in spools)
    assert disk_files and all(file.closed for file, _ in disk_files)
    assert_no_collection(settings.data_dir)
