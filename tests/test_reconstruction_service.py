"""Background reconstruction jobs: queueing, progress, failure isolation and recovery."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
import time
from uuid import uuid4

import pytest

from app.config import Settings
from app.models.reconstruction import (
    ReconstructionJob, ReconstructionRequest, SparseModel, artifact_path_for,
)
from app.modules.reconstruction.base import BackendOutput, ReconstructionBackendError
from app.services.image_collection import ImageCollectionService, ImageInput
from app.services.reconstruction_jobs import ReconstructionJobStore
from app.services.reconstruction_service import ReconstructionService, ReconstructionUnavailable
from app.storage.local_storage import LocalStorage, ProfileNotFoundError, resolve_artifact_path

START = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


class FakeBackend:
    backend_id = "fake"
    backend_version = "fake-1.0"

    def __init__(self, *, sleep=0.0, fail=False, mesh=True, reports=2):
        self.sleep = sleep
        self.fail = fail
        self.mesh = mesh
        self.reports = reports
        self.runs = 0
        self.concurrency = 0
        self.max_concurrency = 0
        self.image_counts = []

    def validate(self, request, profile):
        return []

    def run(self, workspace, request, report, cancel):
        self.runs += 1
        self.concurrency += 1
        self.max_concurrency = max(self.max_concurrency, self.concurrency)
        self.image_counts.append(len(workspace.image_files))
        try:
            report("feature_extraction", 0.2)
            if self.sleep:
                time.sleep(self.sleep)
            report("mapping", 0.6)
            if self.fail:
                raise ReconstructionBackendError("模拟后端失败")
            output = workspace.output_dir
            cameras = output / "cameras.json"
            cameras.write_text("{}", encoding="utf-8")
            cloud = output / "cloud.ply"
            cloud.write_text("ply\n", encoding="utf-8")
            mesh = None
            if self.mesh:
                mesh = output / "mesh.ply"
                mesh.write_text("ply\n", encoding="utf-8")
            return BackendOutput(
                sparse=SparseModel(input_images=len(workspace.image_files),
                                   registered_images=len(workspace.image_files),
                                   points=120, mean_reprojection_error=0.8),
                cameras_file=cameras, point_cloud_file=cloud, mesh_file=mesh,
                point_count=120, mesh_vertices=8 if mesh else None, mesh_faces=12 if mesh else None,
                processing_seconds=0.5,
            )
        finally:
            self.concurrency -= 1


class CancellingBackend(FakeBackend):
    backend_id = "cancelling"

    def run(self, workspace, request, report, cancel):
        for _ in range(500):
            report("mapping", 0.5)
            time.sleep(0.01)
        raise AssertionError("cancel was never observed")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", segmentation_enabled=False, segmentation_backend="baidu",
                    recognition_enabled=False, recognition_api_key="",
                    segmentation_api_key="", segmentation_secret_key="")


@pytest.fixture
def storage(settings):
    return LocalStorage(settings.data_dir)


@pytest.fixture
def profile(settings, png_bytes):
    images = [ImageInput(f"{index}.png", BytesIO(png_bytes)) for index in range(3)]
    return ImageCollectionService(settings).collect("测试物体", images)


def build_service(settings, backend):
    instance = ReconstructionService(settings, backends=(backend,))
    return instance


@pytest.fixture
def service(settings):
    backend = FakeBackend()
    instance = ReconstructionService(settings, backends=(backend,))
    instance.backend = backend
    yield instance
    instance.shutdown()


def request_for(profile, asset_ids=None, backend_id="fake", **options):
    return ReconstructionRequest(
        object_id=profile.object_id,
        asset_ids=asset_ids or [image.asset_id for image in profile.images],
        backend_id=backend_id, options=options,
    )


def test_submit_returns_a_queued_job_and_publishes_the_result(service, storage, profile):
    job = service.submit(request_for(profile))
    assert job.status == "queued"

    finished = service.wait(job.job_id)
    assert finished.status == "succeeded"
    assert finished.progress == 1.0
    assert finished.stage == "export"

    saved, _ = storage.read_profile(profile.object_id)
    assert saved.geometry.model_status == "ready"
    assert saved.geometry.mesh_path is not None
    resolve_artifact_path(storage.data_dir, profile.object_id, saved.geometry.mesh_path, max_bytes=4096, expected_kind="mesh")
    assert saved.geometry.point_cloud_path is not None


def test_workspace_contains_only_the_requested_images(service, profile):
    chosen = [image.asset_id for image in profile.images[:2]]
    job = service.submit(request_for(profile, chosen))
    service.wait(job.job_id)
    assert service.backend.image_counts == [2]


def test_backend_failure_leaves_the_profile_untouched(settings, storage, profile):
    backend = FakeBackend(fail=True)
    service = ReconstructionService(settings, backends=(backend,))
    try:
        job = service.submit(request_for(profile))
        finished = service.wait(job.job_id)
    finally:
        service.shutdown()

    assert finished.status == "failed"
    assert finished.error_code == "reconstruction_failed"
    saved, _ = storage.read_profile(profile.object_id)
    assert saved.geometry.model_status == "not_started"
    assert saved.geometry.mesh_path is None
    assert saved.geometry.point_cloud_path is None


def test_submit_rejects_assets_from_another_object(settings, storage, png_bytes):
    other = ImageCollectionService(settings).collect("另一个", [ImageInput("a.png", BytesIO(png_bytes))])
    backend = FakeBackend()
    service = ReconstructionService(settings, backends=(backend,))
    try:
        collected = ImageCollectionService(settings).collect("本地", [ImageInput("b.png", BytesIO(png_bytes))])
        with pytest.raises(ReconstructionUnavailable, match="不属于该物体"):
            service.submit(request_for(collected, [other.images[0].asset_id]))
    finally:
        service.shutdown()


def test_submit_rejects_an_unknown_backend_and_object(service, profile):
    with pytest.raises(ReconstructionUnavailable, match="未注册的重建后端"):
        service.submit(request_for(profile, backend_id="colmap"))

    missing = ReconstructionRequest(object_id=uuid4(),
                                    asset_ids=[image.asset_id for image in profile.images],
                                    backend_id="fake")
    with pytest.raises(ProfileNotFoundError):
        service.submit(missing)


def test_queued_job_can_be_cancelled_before_it_runs(settings, profile):
    slow = FakeBackend(sleep=0.3)
    service = ReconstructionService(settings, backends=(slow,))
    try:
        first = service.submit(request_for(profile))
        second = service.submit(request_for(profile))
        canceled = service.cancel(second.job_id)
        assert canceled.status == "canceled"
        service.wait(first.job_id)
        assert slow.runs == 1
        assert service.get(second.job_id).status == "canceled"
    finally:
        service.shutdown()


def test_running_job_stops_at_the_next_stage_boundary(settings, storage, profile):
    backend = CancellingBackend()
    service = ReconstructionService(settings, backends=(backend,))
    try:
        job = service.submit(request_for(profile, backend_id="cancelling"))
        deadline = time.monotonic() + 5
        while service.get(job.job_id).status != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        service.cancel(job.job_id)
        finished = service.wait(job.job_id)
    finally:
        service.shutdown()

    assert finished.status == "canceled"
    saved, _ = storage.read_profile(profile.object_id)
    assert saved.geometry.model_status == "not_started"


def test_timeout_marks_the_job_failed(settings, storage, profile):
    backend = FakeBackend(sleep=0.3)
    service = ReconstructionService(settings, backends=(backend,), job_timeout_seconds=0.05)
    try:
        job = service.submit(request_for(profile))
        finished = service.wait(job.job_id)
    finally:
        service.shutdown()

    assert finished.status == "failed"
    assert finished.error_code == "reconstruction_timeout"
    saved, _ = storage.read_profile(profile.object_id)
    assert saved.geometry.model_status == "not_started"


def test_recover_marks_running_and_queued_records(settings, storage, profile):
    store = ReconstructionJobStore(settings.data_dir)
    running = ReconstructionJob(object_id=profile.object_id, backend_id="fake", status="running",
                                stage="mapping", progress=0.4, input_images=1,
                                created_at=START, started_at=START)
    queued = ReconstructionJob(object_id=profile.object_id, backend_id="fake", input_images=1,
                               created_at=START + timedelta(seconds=1))
    store.save(running)
    store.save(queued)

    service = ReconstructionService(settings, backends=(FakeBackend(),))
    try:
        recovered = {job.job_id: job for job in service.recover()}
    finally:
        service.shutdown()

    assert recovered[running.job_id].status == "interrupted"
    assert recovered[running.job_id].error_code == "service_restarted"
    assert recovered[queued.job_id].status == "canceled"
    assert recovered[running.job_id].finished_at is not None


def test_sparse_only_run_is_published_as_partial(storage, profile, settings):
    backend = FakeBackend(mesh=False)
    service = ReconstructionService(settings, backends=(backend,))
    try:
        job = service.submit(request_for(profile))
        finished = service.wait(job.job_id)
    finally:
        service.shutdown()

    assert finished.status == "succeeded"
    saved, _ = storage.read_profile(profile.object_id)
    assert saved.geometry.model_status == "partial"
    assert saved.geometry.mesh_path is None
    assert saved.geometry.point_cloud_path is not None
    assert any("网格" in note for note in saved.geometry.quality_notes)


def test_jobs_run_one_at_a_time(settings, profile):
    backend = FakeBackend(sleep=0.05)
    service = ReconstructionService(settings, backends=(backend,))
    try:
        first = service.submit(request_for(profile))
        second = service.submit(request_for(profile))
        service.wait(first.job_id, timeout=10)
        service.wait(second.job_id, timeout=10)
    finally:
        service.shutdown()

    assert backend.max_concurrency == 1
    assert backend.runs == 2


def test_workspace_is_removed_after_the_job(settings, storage, profile):
    backend = FakeBackend()
    service = ReconstructionService(settings, backends=(backend,))
    try:
        job = service.submit(request_for(profile))
        service.wait(job.job_id)
    finally:
        service.shutdown()

    staging = storage.data_dir / ".staging"
    leftovers = [path for path in staging.iterdir()] if staging.exists() else []
    assert leftovers == []


def test_job_record_persists_on_disk(service, storage, profile):
    job = service.submit(request_for(profile))
    service.wait(job.job_id)
    record = storage.data_dir / "reconstruction_jobs" / f"{job.job_id}.json"
    assert record.is_file()
    reloaded = ReconstructionJob.model_validate_json(record.read_bytes())
    assert reloaded.status == "succeeded"


def test_artifact_paths_stay_inside_the_object(service, storage, profile):
    job = service.submit(request_for(profile))
    service.wait(job.job_id)
    saved, _ = storage.read_profile(profile.object_id)
    prefix = f"objects/{profile.object_id}/geometry/"
    assert saved.geometry.mesh_path.startswith(prefix)
    assert saved.geometry.point_cloud_path.startswith(prefix)
