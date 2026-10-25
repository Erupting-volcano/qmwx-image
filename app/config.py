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

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        for name in ("max_image_bytes", "max_files", "max_total_bytes", "max_image_pixels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
