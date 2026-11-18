"""Backend contract: receive a prepared workspace, return owned files, never touch storage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from app.models.object_profile import ObjectProfile
from app.models.reconstruction import CameraIntrinsics, ReconstructionRequest, SparseModel


class ReconstructionBackendError(RuntimeError):
    """A safe, user-facing explanation for a failed backend run."""


class ReconstructionCancelled(RuntimeError):
    """Raised by the progress reporter when the job was cancelled or timed out."""


@dataclass(frozen=True)
class ReconstructionWorkspace:
    """Scratch space owned by the service and deleted after the job."""

    root: Path
    images_dir: Path
    masks_dir: Path
    output_dir: Path
    image_files: tuple[Path, ...] = ()
    mask_files: tuple[Path, ...] = ()
    options: dict = None

    def __post_init__(self) -> None:
        if self.options is None:
            object.__setattr__(self, "options", {})


@dataclass(frozen=True)
class BackendOutput:
    """Results of a run; paths must live inside the workspace output directory."""

    sparse: SparseModel | None = None
    intrinsics: CameraIntrinsics | None = None
    cameras_file: Path | None = None
    point_cloud_file: Path | None = None
    mesh_file: Path | None = None
    point_count: int | None = None
    mesh_vertices: int | None = None
    mesh_faces: int | None = None
    processing_seconds: float | None = None
    warnings: tuple[str, ...] = ()


ProgressReporter = Callable[[str, float], None]
CancelCheck = Callable[[], bool]


@runtime_checkable
class ReconstructionBackend(Protocol):
    backend_id: str
    backend_version: str

    def validate(self, request: ReconstructionRequest, profile: ObjectProfile) -> list[str]:
        """Return blocking, user-safe problems; never contact the engine here."""

    def run(self, workspace: ReconstructionWorkspace, request: ReconstructionRequest,
            report: ProgressReporter, cancel: CancelCheck) -> BackendOutput:
        """Blocking call; report stages through the reporter so the job stays visible."""
