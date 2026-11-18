"""HTTP surface for reconstruction: submit, poll, cancel, download geometry."""

from io import BytesIO
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.reconstruction import SparseModel
from app.modules.reconstruction.base import BackendOutput
from app.services.image_collection import ImageCollectionService, ImageInput
from app.services.reconstruction_service import ReconstructionService


class FakeBackend:
    backend_id = "fake"
    backend_version = "fake-1.0"

    def __init__(self, *, sleep=0.0):
        self.sleep = sleep

    def validate(self, request, profile):
        return []

    def run(self, workspace, request, report, cancel):
        report("mapping", 0.5)
        if self.sleep:
            time.sleep(self.sleep)
        output = workspace.output_dir
        cameras = output / "cameras.json"
        cameras.write_text("{}", encoding="utf-8")
        cloud = output / "cloud.ply"
        cloud.write_bytes(b"ply\n")
        return BackendOutput(sparse=SparseModel(input_images=len(workspace.image_files),
                                                registered_images=len(workspace.image_files),
                                                points=10, mean_reprojection_error=0.5),
                             cameras_file=cameras, point_cloud_file=cloud, point_count=10)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", reconstruction_enabled=False,
                    segmentation_enabled=False, recognition_enabled=False)


@pytest.fixture
def collected(settings, png_bytes):
    images = [ImageInput(f"{index}.png", BytesIO(png_bytes)) for index in range(3)]
    return ImageCollectionService(settings).collect("测试物体", images)


def make_client(settings, backend=None):
    if backend is None:
        return TestClient(create_app(settings))
    service = ReconstructionService(settings, backends=(backend,))
    return TestClient(create_app(settings, reconstruction_service=service))


def wait_for(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/reconstructions/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed", "canceled", "interrupted"}:
            return body
        time.sleep(0.02)
    raise AssertionError("任务未在超时内结束")


def test_submit_poll_and_download_geometry(settings, collected):
    with make_client(settings, FakeBackend()) as client:
        response = client.post(f"/api/v1/objects/{collected.object_id}/reconstructions", json={})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert response.json()["status"] == "queued"

        body = wait_for(client, job_id)
        assert body["status"] == "succeeded"

        profile = client.get(f"/api/v1/objects/{collected.object_id}").json()
        assert profile["geometry"]["point_cloud_path"] is not None
        assert profile["geometry"]["model_status"] == "partial"

        artifact = client.get(f"/api/v1/objects/{collected.object_id}/geometry/point_cloud")
        assert artifact.status_code == 200
        assert artifact.content == b"ply\n"


def test_reconstruction_is_disabled_without_a_service(settings, collected):
    with make_client(settings) as client:
        response = client.post(f"/api/v1/objects/{collected.object_id}/reconstructions", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "reconstruction_disabled"


def test_unknown_object_and_job_are_404(settings, collected):
    with make_client(settings, FakeBackend()) as client:
        missing = client.post(f"/api/v1/objects/{uuid4()}/reconstructions", json={})
        assert missing.status_code == 404
        assert client.get(f"/api/v1/reconstructions/{uuid4()}").status_code == 404


def test_foreign_assets_are_rejected(settings, collected):
    with make_client(settings, FakeBackend()) as client:
        response = client.post(f"/api/v1/objects/{collected.object_id}/reconstructions",
                               json={"asset_ids": [str(uuid4())]})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "reconstruction_unavailable"


def test_missing_artifact_is_404(settings, collected):
    with make_client(settings, FakeBackend()) as client:
        response = client.get(f"/api/v1/objects/{collected.object_id}/geometry/point_cloud")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "artifact_not_found"
        assert client.get(f"/api/v1/objects/{collected.object_id}/geometry/nonsense").status_code == 404


def test_cancel_queued_job(settings, collected):
    backend = FakeBackend(sleep=0.3)
    with make_client(settings, backend) as client:
        first = client.post(f"/api/v1/objects/{collected.object_id}/reconstructions", json={}).json()
        second = client.post(f"/api/v1/objects/{collected.object_id}/reconstructions", json={}).json()
        canceled = client.post(f"/api/v1/reconstructions/{second['job_id']}/cancel")
        assert canceled.status_code == 200
        assert canceled.json()["status"] == "canceled"
        wait_for(client, first["job_id"])
