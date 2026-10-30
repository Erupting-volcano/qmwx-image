"""Small, explicit runtime configuration without an extra settings dependency."""

from dataclasses import dataclass, field
import math
import json
import os
from pathlib import Path
import re


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


def _env_json_object(name: str) -> dict:
    try:
        value = json.loads(os.environ.get(name, "{}"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a JSON object") from None


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
    segmentation_backend: str = field(default_factory=lambda: os.environ.get(
        "DIGITIZER_SEGMENTATION_BACKEND", "baidu"
    ).strip())
    segmentation_api_key: str = field(default_factory=lambda: os.environ.get(
        "BAIDU_SEGMENTATION_API_KEY", ""
    ).strip(), repr=False)
    segmentation_secret_key: str = field(default_factory=lambda: os.environ.get(
        "BAIDU_SEGMENTATION_SECRET_KEY", ""
    ).strip(), repr=False)
    segmentation_timeout_seconds: float = field(default_factory=lambda: float(os.environ.get(
        "DIGITIZER_SEGMENTATION_TIMEOUT_SECONDS", "30"
    )))
    # Interfaces for the future local adapter; no existence checks or imports at startup.
    sam2_checkpoint_path: Path | None = field(default_factory=lambda: (
        Path(os.environ["SAM2_CHECKPOINT_PATH"].strip())
        if os.environ.get("SAM2_CHECKPOINT_PATH", "").strip() else None
    ), repr=False)
    sam2_device: str = field(default_factory=lambda: os.environ.get("SAM2_DEVICE", "auto").strip().lower())
    recognition_enabled: bool = field(default_factory=lambda: _env_bool("DIGITIZER_RECOGNITION_ENABLED"))
    recognition_combine_tasks: bool = field(default_factory=lambda: _env_bool("DIGITIZER_RECOGNITION_COMBINE_TASKS", True))
    recognition_backend: str = field(default_factory=lambda: os.environ.get(
        "DIGITIZER_RECOGNITION_BACKEND", "openai_compatible").strip())
    recognition_model: str = field(default_factory=lambda: os.environ.get(
        "DIGITIZER_RECOGNITION_MODEL", "qwen3-vl-plus").strip())
    recognition_api_base: str = field(default_factory=lambda: os.environ.get(
        "DIGITIZER_RECOGNITION_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(), repr=False)
    recognition_api_key: str = field(default_factory=lambda: os.environ.get(
        "DIGITIZER_RECOGNITION_API_KEY", "").strip(), repr=False)
    recognition_timeout: float = field(default_factory=lambda: float(os.environ.get("DIGITIZER_RECOGNITION_TIMEOUT", "60")))
    recognition_max_images: int = field(default_factory=lambda: _env_int("DIGITIZER_RECOGNITION_MAX_IMAGES", 10))
    recognition_max_image_side: int = field(default_factory=lambda: _env_int("DIGITIZER_RECOGNITION_MAX_IMAGE_SIDE", 1536))
    recognition_max_concurrency: int = field(default_factory=lambda: _env_int("DIGITIZER_RECOGNITION_MAX_CONCURRENCY", 2))
    recognition_min_interval: float = field(default_factory=lambda: float(os.environ.get("DIGITIZER_RECOGNITION_MIN_INTERVAL", "0.25")))
    recognition_max_output_tokens: int = field(default_factory=lambda: _env_int("DIGITIZER_RECOGNITION_MAX_OUTPUT_TOKENS", 8192))
    recognition_task_models: dict = field(default_factory=lambda: _env_json_object("DIGITIZER_RECOGNITION_TASK_MODELS"))
    recognition_model_configs: dict = field(default_factory=lambda: _env_json_object("DIGITIZER_RECOGNITION_MODEL_CONFIGS"), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        for name in ("max_image_bytes", "max_files", "max_total_bytes", "max_image_pixels"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.segmentation_enabled, bool):
            raise ValueError("segmentation_enabled must be a boolean")
        if (not isinstance(self.segmentation_backend, str)
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", self.segmentation_backend)):
            raise ValueError("segmentation_backend must be a model identifier")
        if self.sam2_checkpoint_path is not None:
            object.__setattr__(self, "sam2_checkpoint_path", Path(self.sam2_checkpoint_path).expanduser())
        if not isinstance(self.sam2_device, str):
            raise ValueError("sam2_device must be text")
        for name in ("segmentation_api_key", "segmentation_secret_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError("Segmentation API credentials must be text without control characters")
        timeout = self.segmentation_timeout_seconds
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 1 <= timeout <= 120):
            raise ValueError("segmentation_timeout_seconds must be between 1 and 120")
        if type(self.recognition_enabled) is not bool or type(self.recognition_combine_tasks) is not bool:
            raise ValueError("recognition_enabled and recognition_combine_tasks must be booleans")
        for name in ("recognition_backend", "recognition_model"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", getattr(self, name)):
                raise ValueError(f"{name} must be an identifier")
        if not isinstance(self.recognition_api_base, str) or not isinstance(self.recognition_api_key, str):
            raise ValueError("Recognition endpoint and credentials must be text")
        if any(ord(char) < 32 or ord(char) == 127 for char in self.recognition_api_key):
            raise ValueError("Recognition credentials cannot contain control characters")
        for name, upper in (("recognition_max_images", 20), ("recognition_max_image_side", 4096),
                            ("recognition_max_concurrency", 8), ("recognition_max_output_tokens", 32768)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} is outside its supported positive range")
        for name, lower, upper in (("recognition_timeout", 1, 300), ("recognition_min_interval", 0, 60)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} is outside its finite range")
        for name in ("recognition_task_models", "recognition_model_configs"):
            value = getattr(self, name)
            if not isinstance(value, dict) or len(value) > 100:
                raise ValueError(f"{name} must be a bounded mapping")
            if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", key) for key in value):
                raise ValueError(f"{name} contains invalid identifiers")
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value)
               for value in self.recognition_task_models.values()):
            raise ValueError("Recognition task routes must contain model identifiers")
        for config in self.recognition_model_configs.values():
            if (not isinstance(config, dict) or set(config) - {"api_base", "model", "api_key_env", "timeout"}
                    or any(not isinstance(config.get(key, ""), str) for key in ("api_base", "model", "api_key_env"))):
                raise ValueError("Recognition model configuration accepts endpoint/model/key environment name/timeout only")
            if "api_key_env" in config and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", config["api_key_env"]):
                raise ValueError("Recognition credential source must be an environment variable name")
        # Detach caller mappings; frozen settings must not share mutable route input.
        object.__setattr__(self, "recognition_task_models", dict(self.recognition_task_models))
        object.__setattr__(self, "recognition_model_configs", json.loads(json.dumps(self.recognition_model_configs)))
