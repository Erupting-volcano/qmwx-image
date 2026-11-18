"""One JSON record per reconstruction job; atomic writes, restart-safe reads."""

import json
import os
from pathlib import Path
import threading
import time
from uuid import UUID, uuid4

from app.models.reconstruction import ReconstructionJob
from app.storage.local_storage import storage_io_path


REPLACE_ATTEMPTS = 20
REPLACE_DELAY = 0.01


def _replace_record(source: Path, destination: Path) -> None:
    """Publish a record even while another thread is reading the previous one.

    Windows refuses os.replace() while a reader holds the destination without
    delete sharing, which is exactly what the polling endpoint does.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(storage_io_path(source), storage_io_path(destination))
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_DELAY)


class JobNotFoundError(Exception):
    """The requested job id has no stored record."""


class JobStoreError(Exception):
    """A stored record exists but cannot be trusted."""


class ReconstructionJobStore:
    """File-backed job records under data/reconstruction_jobs."""

    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "reconstruction_jobs"
        # Windows denies reading a file while it is being replaced; serialise
        # readers and writers instead of racing the operating system.
        self._lock = threading.RLock()

    def record_path(self, job_id: UUID) -> Path:
        return self.root / f"{UUID(str(job_id))}.json"

    def save(self, job: ReconstructionJob) -> None:
        payload = job.model_dump(mode="json")
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with self._lock:
            storage_io_path(self.root).mkdir(parents=True, exist_ok=True)
            temporary = self.root / f".{job.job_id}.{uuid4()}.tmp"
            with storage_io_path(temporary).open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_record(temporary, self.record_path(job.job_id))

    def get(self, job_id: UUID) -> ReconstructionJob:
        path = self.record_path(job_id)
        with self._lock:
            try:
                content = storage_io_path(path).read_bytes()
            except FileNotFoundError as exc:
                raise JobNotFoundError("重建任务不存在") from exc
        try:
            job = ReconstructionJob.model_validate_json(content)
        except (ValueError, TypeError) as exc:
            raise JobStoreError("重建任务记录损坏") from exc
        if job.job_id != UUID(str(job_id)):
            raise JobStoreError("重建任务记录与编号不一致")
        return job

    def list(self) -> list[ReconstructionJob]:
        if not storage_io_path(self.root).is_dir():
            return []
        jobs = []
        with self._lock:
            entries = list(storage_io_path(self.root).glob("*.json"))
        for entry in entries:
            try:
                jobs.append(ReconstructionJob.model_validate_json(entry.read_bytes()))
            except (ValueError, TypeError, OSError):
                # A corrupt or half-written record must not hide the healthy ones.
                continue
        return sorted(jobs, key=lambda job: job.created_at)
