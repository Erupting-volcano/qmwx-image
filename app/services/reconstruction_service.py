"""Background reconstruction jobs: queue, progress, recovery; storage stays here, not in engines."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import UUID, uuid4

from app.config import Settings
from app.models.object_profile import GeometryInfo, ObjectProfile
from app.models.reconstruction import (
    ReconstructionJob, ReconstructionRequest, ReconstructionResult,
)
from app.modules.reconstruction.base import (
    BackendOutput, ReconstructionBackend, ReconstructionCancelled, ReconstructionWorkspace,
)
from app.services.reconstruction_jobs import JobNotFoundError, ReconstructionJobStore
from app.storage.local_storage import (
    LocalStorage, ProfileNotFoundError, resolve_mask_path, resolve_original_path, storage_io_path,
)


logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "canceled", "interrupted"})
PROGRESS_WRITE_INTERVAL = 2.0


class ReconstructionUnavailable(RuntimeError):
    """A safe, user-facing reason why this request cannot start."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReconstructionService:
    """Submit work, keep job records current, publish results through the update transaction."""

    def __init__(self, settings: Settings | None = None, *, storage: LocalStorage | None = None,
                 store: ReconstructionJobStore | None = None,
                 backends: tuple[ReconstructionBackend, ...] = (),
                 max_workers: int = 1, job_timeout_seconds: float = 3600.0,
                 max_artifact_bytes: int = 512 * 1024 * 1024):
        self.settings = settings or Settings()
        self.storage = storage or LocalStorage(self.settings.data_dir)
        self.store = store or ReconstructionJobStore(self.settings.data_dir)
        self.backends = {backend.backend_id: backend for backend in backends}
        if len(self.backends) != len(backends):
            raise ValueError("重建后端标识不得重复")
        if max_workers != 1:
            # Two COLMAP runs thrash the machine and could update one object twice.
            raise ValueError("重建服务只允许一个工作线程")
        self.job_timeout_seconds = job_timeout_seconds
        self.max_artifact_bytes = max_artifact_bytes
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reconstruction")
        self._cancel_events: dict[UUID, threading.Event] = {}
        self._lock = threading.Lock()

    # -- public API -------------------------------------------------------

    def submit(self, request: ReconstructionRequest) -> ReconstructionJob:
        """Record the job and start it; returns immediately with a queued record."""
        backend = self.backends.get(request.backend_id)
        if backend is None:
            raise ReconstructionUnavailable(f"未注册的重建后端：{request.backend_id}")
        profile, _ = self.storage.read_profile(request.object_id)
        known = {image.asset_id for image in profile.images}
        if any(asset_id not in known for asset_id in request.asset_ids):
            raise ReconstructionUnavailable("重建请求包含不属于该物体的图片")
        problems = backend.validate(request, profile)
        if problems:
            raise ReconstructionUnavailable("；".join(problems))

        job = ReconstructionJob(object_id=request.object_id, backend_id=backend.backend_id,
                                input_images=len(request.asset_ids), created_at=_now())
        self.store.save(job)
        event = threading.Event()
        with self._lock:
            self._cancel_events[job.job_id] = event
        self._executor.submit(self._execute, job.job_id, request, event)
        return job

    def get(self, job_id: UUID) -> ReconstructionJob:
        return self.store.get(job_id)

    def cancel(self, job_id: UUID) -> ReconstructionJob:
        """Cancel a queued job immediately; ask a running job to stop at a stage boundary."""
        job = self.store.get(job_id)
        if job.status in TERMINAL_STATUSES:
            return job
        with self._lock:
            event = self._cancel_events.get(job.job_id)
        if event is not None:
            event.set()
        if job.status == "queued":
            job = self._transition(job, status="canceled", finished_at=_now(),
                                  error_message="任务在开始前被取消")
            self.store.save(job)
        return job

    def recover(self) -> list[ReconstructionJob]:
        """Mark records left behind by a previous process; nothing is resumed."""
        recovered = []
        for job in self.store.list():
            if job.status == "running":
                updated = self._transition(job, status="interrupted", finished_at=_now(),
                                           error_code="service_restarted",
                                           error_message="服务重启导致任务中断，不会自动续跑")
            elif job.status == "queued":
                updated = self._transition(job, status="canceled", finished_at=_now(),
                                           error_message="服务重启前任务尚未开始")
            else:
                continue
            self.store.save(updated)
            recovered.append(updated)
        return recovered

    def wait(self, job_id: UUID, timeout: float = 30.0) -> ReconstructionJob:
        """Poll a job until it reaches a terminal status; used by callers and tests."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.store.get(job_id)
            if job.status in TERMINAL_STATUSES:
                return job
            time.sleep(0.01)
        raise TimeoutError("等待重建任务超时")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- execution --------------------------------------------------------

    def _execute(self, job_id: UUID, request: ReconstructionRequest, cancel_event: threading.Event) -> None:
        workspace = None
        try:
            job = self.store.get(job_id)
            if job.status != "queued":
                return
            job = self._transition(job, status="running", stage="preflight", progress=0.01,
                                   started_at=_now(), finished_at=None)
            self.store.save(job)
            state = {"job": job}
            last_write = [0.0]
            deadline = time.monotonic() + self.job_timeout_seconds

            def report(stage: str, progress: float) -> None:
                if cancel_event.is_set():
                    raise ReconstructionCancelled("cancelled")
                if time.monotonic() > deadline:
                    raise ReconstructionCancelled("timeout")
                now = time.monotonic()
                if stage != state["job"].stage or now - last_write[0] >= PROGRESS_WRITE_INTERVAL:
                    updated = self._transition(state["job"], status="running", stage=stage,
                                               progress=min(max(progress, 0.0), 0.99))
                    self.store.save(updated)
                    state["job"] = updated
                    last_write[0] = now

            backend = self.backends[request.backend_id]
            workspace = self._prepare_workspace(job, request)
            output = backend.run(workspace, request, report, cancel_event.is_set)
            if cancel_event.is_set():
                raise ReconstructionCancelled("cancelled")
            self._publish(job, request, output)
            self.store.save(self._transition(state["job"], status="succeeded", stage="export",
                                             progress=1.0, finished_at=_now()))
        except ReconstructionCancelled as exc:
            timeout = str(exc) == "timeout"
            job = self.store.get(job_id)
            if job.status not in TERMINAL_STATUSES:
                self.store.save(self._transition(
                    job,
                    status="failed" if timeout else "canceled",
                    finished_at=_now(),
                    error_code="reconstruction_timeout" if timeout else None,
                    error_message="重建超时" if timeout else "任务已取消",
                ))
        except Exception:
            error_id = uuid4()
            logger.exception("Reconstruction job failed job=%s error_id=%s", job_id, error_id)
            try:
                job = self.store.get(job_id)
                if job.status not in TERMINAL_STATUSES:
                    self.store.save(self._transition(
                        job, status="failed", finished_at=_now(), error_code="reconstruction_failed",
                        error_message=f"重建执行失败，请凭错误编号排查（{error_id}）",
                    ))
            except Exception:
                logger.exception("Could not record reconstruction failure job=%s", job_id)
        finally:
            if workspace is not None:
                self._cleanup_workspace(workspace.root)
            with self._lock:
                self._cancel_events.pop(job_id, None)

    def _prepare_workspace(self, job: ReconstructionJob, request: ReconstructionRequest) -> ReconstructionWorkspace:
        root = self.storage.data_dir / ".staging" / f"reconstruction-{job.job_id}"
        images_dir = root / "images"
        masks_dir = root / "masks"
        output_dir = root / "output"
        for directory in (images_dir, masks_dir, output_dir):
            storage_io_path(directory).mkdir(parents=True, exist_ok=True)

        profile, _ = self.storage.read_profile(job.object_id)
        assets = {image.asset_id: image for image in profile.images}
        masks = self._masks_by_asset(profile)
        image_files: list[Path] = []
        mask_files: list[Path] = []
        for index, asset_id in enumerate(request.asset_ids):
            asset = assets[asset_id]
            source = resolve_original_path(self.storage.data_dir, job.object_id, asset_id, asset.storage_path)
            target = images_dir / f"{index:03d}{source.suffix.lower()}"
            self._link(source, target)
            image_files.append(target)
            if request.use_masks and asset_id in masks:
                mask_target = masks_dir / f"{index:03d}.png"
                self._link(masks[asset_id], mask_target)
                mask_files.append(mask_target)
        return ReconstructionWorkspace(
            root=root, images_dir=images_dir, masks_dir=masks_dir, output_dir=output_dir,
            image_files=tuple(image_files), mask_files=tuple(mask_files), options=dict(request.options),
        )

    def _masks_by_asset(self, profile: ObjectProfile) -> dict:
        """Latest usable segmentation masks, validated the same way previews are."""
        from app.models.segmentation import SegmentationData

        resolved = {}
        for result in reversed(profile.processing_results):
            if result.module_name != "object_segmentation" or result.status not in {"success", "partial"}:
                continue
            try:
                payload = SegmentationData.model_validate(result.data)
            except (ValueError, TypeError):
                continue
            for image in payload.images:
                if image.mask_path is None or image.asset_id in resolved:
                    continue
                try:
                    resolved[image.asset_id] = resolve_mask_path(
                        self.storage.data_dir, profile.object_id, image.asset_id, image.mask_path,
                        expected_size=(image.width, image.height),
                        max_pixels=self.settings.max_image_pixels,
                        expected_bbox=image.bbox, expected_area_ratio=image.area_ratio,
                    )
                except (ValueError, TypeError, OSError):
                    continue
            break
        return resolved

    def _publish(self, job: ReconstructionJob, request: ReconstructionRequest, output: BackendOutput) -> ReconstructionResult:
        """Copy artifacts into the archive and write the geometry update atomically."""
        with self.storage.update_transaction(job.object_id) as transaction:
            cameras_path = self._store_artifact(transaction, "cameras", output.cameras_file)
            point_cloud_path = self._store_artifact(transaction, "point_cloud", output.point_cloud_file)
            mesh_path = self._store_artifact(transaction, "mesh", output.mesh_file)
            warnings = list(output.warnings)
            status = "succeeded" if mesh_path is not None else "partial"
            if status == "partial":
                warnings.append("仅完成稀疏或点云重建，尚未生成网格")
            result = ReconstructionResult(
                job_id=job.job_id, object_id=job.object_id, backend_id=job.backend_id,
                backend_version=output_backend_version(self.backends[job.backend_id]),
                status=status, input_images=len(request.asset_ids),
                sparse=output.sparse, intrinsics=output.intrinsics,
                cameras_path=cameras_path, point_cloud_path=point_cloud_path, mesh_path=mesh_path,
                point_count=output.point_count, mesh_vertices=output.mesh_vertices,
                mesh_faces=output.mesh_faces, processing_seconds=output.processing_seconds,
                warnings=list(dict.fromkeys(warnings)),
            )
            profile = transaction.read_profile()
            fields = profile.geometry.model_dump()
            update = result.to_geometry_update()
            fields.update(update)
            fields["quality_notes"] = list(dict.fromkeys([*fields.get("quality_notes", []), *update["quality_notes"]]))
            profile.geometry = GeometryInfo.model_validate(fields)
            transaction.save_profile(profile)
            transaction.commit()
        return result

    def _store_artifact(self, transaction, kind: str, source: Path | None) -> str | None:
        if source is None:
            return None
        return transaction.save_artifact(kind, source, max_bytes=self.max_artifact_bytes)

    @staticmethod
    def _link(source: Path, target: Path) -> None:
        try:
            os.link(storage_io_path(source), storage_io_path(target))
        except OSError:
            shutil.copy2(storage_io_path(source), storage_io_path(target))

    def _cleanup_workspace(self, root: Path) -> None:
        allowed = (self.storage.data_dir / ".staging").resolve()
        target = root.resolve()
        if target == allowed or not target.is_relative_to(allowed):
            return
        shutil.rmtree(storage_io_path(root), ignore_errors=True)

    @staticmethod
    def _transition(job: ReconstructionJob, **updates) -> ReconstructionJob:
        payload = job.model_dump(mode="json")
        payload.update(updates)
        return ReconstructionJob.model_validate(payload)


def output_backend_version(backend: ReconstructionBackend) -> str | None:
    version = getattr(backend, "backend_version", None)
    return version if isinstance(version, str) and version.strip() else None
