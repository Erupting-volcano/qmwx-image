"""Small, explicit runtime configuration without an extra settings dependency."""

from dataclasses import dataclass, field
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("DIGITIZER_DATA_DIR", str(PROJECT_ROOT / "data"))
    ))
    max_image_bytes: int = field(default_factory=lambda: _env_int(
        "DIGITIZER_MAX_IMAGE_BYTES", 15 * 1024 * 1024
    ))
    max_files: int = field(default_factory=lambda: _env_int("DIGITIZER_MAX_FILES", 20))
    max_total_bytes: int = field(default_factory=lambda: _env_int(
        "DIGITIZER_MAX_TOTAL_BYTES", 100 * 1024 * 1024
    ))
    max_image_pixels: int = field(default_factory=lambda: _env_int(
        "DIGITIZER_MAX_IMAGE_PIXELS", 40_000_000
    ))
    segmentation_enabled: bool = field(default_factory=lambda: _env_bool("DIGITIZER_SEGMENTATION_ENABLED"))
    segmentation_checkpoint: Path | None = field(default_factory=lambda: (
        Path(os.environ["DIGITIZER_SEGMENTATION_CHECKPOINT"])
        if os.environ.get("DIGITIZER_SEGMENTATION_CHECKPOINT", "").strip() else None
    ))
    segmentation_device: str = field(default_factory=lambda: os.environ.get(
        "DIGITIZER_SEGMENTATION_DEVICE", "cpu"
    ).strip().lower())

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        for name in ("max_image_bytes", "max_files", "max_total_bytes", "max_image_pixels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.segmentation_enabled, bool):
            raise ValueError("segmentation_enabled must be a boolean")
        if self.segmentation_device not in {"cpu", "cuda"}:
            raise ValueError("segmentation_device must be cpu or cuda")
        if self.segmentation_checkpoint is not None:
            object.__setattr__(self, "segmentation_checkpoint",
                               Path(self.segmentation_checkpoint).expanduser().resolve())
