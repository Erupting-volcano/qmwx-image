"""Model-neutral VLM contracts. Image pixels and provider secrets never enter JSON."""

from dataclasses import dataclass, field, fields
import json
import math
import re
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from PIL import Image


class RecognitionError(RuntimeError):
    """A fixed safe explanation, suitable for an API response or log."""


class RecognitionUnavailable(RecognitionError):
    """Missing configuration, an unregistered model, or failed initialization."""


class UnsupportedRecognitionCapability(RecognitionUnavailable):
    """The adapter does not implement a capability required by the task."""


def valid_identifier(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value) is not None


def bounded_json(value: object, *, max_bytes: int = 262144) -> dict:
    if not isinstance(value, dict):
        raise ValueError("识别数据必须为JSON对象")

    def check(item, depth=0):
        if depth > 16:
            raise ValueError("识别JSON层级过深")
        if item is None or type(item) in {str, bool, int, float}:
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            if len(item) > 10000:
                raise ValueError("识别JSON对象过大")
            for child in item.values():
                check(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            if len(item) > 10000:
                raise ValueError("识别JSON列表过长")
            for child in item:
                check(child, depth + 1)
        else:
            raise ValueError("识别JSON不能包含图像、二进制或模型对象")

    check(value)
    try:
        content = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, OverflowError, TypeError):
        raise ValueError("识别JSON含无效数值或类型") from None
    if len(content.encode("utf-8")) > max_bytes:
        raise ValueError("识别JSON超过安全大小限制")
    return json.loads(content)


@dataclass(frozen=True)
class RecognitionCapabilities:
    image_input: bool = False
    multi_image_input: bool = False
    structured_output: bool = False
    json_schema: bool = False
    high_resolution_image: bool = False
    multi_turn: bool = False
    video_input: bool = False

    def __post_init__(self):
        if any(type(getattr(self, item.name)) is not bool for item in fields(self)):
            raise ValueError("视觉模型能力声明必须为布尔值")


@dataclass(frozen=True)
class RecognitionImage:
    asset_id: UUID
    image: Image.Image = field(repr=False)
    role: Literal["original", "target_region"] = "original"

    def __post_init__(self):
        object.__setattr__(self, "asset_id", UUID(str(self.asset_id)))
        if (not isinstance(self.image, Image.Image) or self.image.mode != "RGB" or min(self.image.size) <= 0
                or self.role not in {"original", "target_region"}):
            raise ValueError("视觉输入必须为已转正的RGB图像，并明确原图或目标区域角色")


@dataclass(frozen=True)
class RecognitionRequest:
    task_id: str
    instruction: str = field(repr=False)
    images: tuple[RecognitionImage, ...] = field(default=(), repr=False)
    output_schema: dict = field(default_factory=dict, repr=False)
    context: dict = field(default_factory=dict, repr=False)
    required_capabilities: RecognitionCapabilities = field(
        default_factory=lambda: RecognitionCapabilities(structured_output=True),
    )

    def __post_init__(self):
        if not valid_identifier(self.task_id):
            raise ValueError("识别任务标识无效")
        if not isinstance(self.instruction, str) or not self.instruction.strip() or len(self.instruction) > 65536:
            raise ValueError("识别指令必须为有界的非空文本")
        if (not isinstance(self.images, tuple) or len(self.images) > 40
                or any(not isinstance(image, RecognitionImage) for image in self.images)):
            raise ValueError("识别图片必须为有界的标准图片元组")
        for image in self.images:
            image.__post_init__()
        if not isinstance(self.required_capabilities, RecognitionCapabilities):
            raise ValueError("识别请求必须声明标准模型能力需求")
        object.__setattr__(self, "output_schema", bounded_json(self.output_schema))
        object.__setattr__(self, "context", bounded_json(self.context))


def check_recognition_capabilities(request: RecognitionRequest, capabilities: RecognitionCapabilities) -> None:
    if not isinstance(request, RecognitionRequest) or not isinstance(capabilities, RecognitionCapabilities):
        raise TypeError("需要标准识别请求和模型能力声明")
    request.__post_init__()
    required = request.required_capabilities
    if any(getattr(required, item.name) and not getattr(capabilities, item.name) for item in fields(required)):
        raise UnsupportedRecognitionCapability("所选视觉模型不满足任务要求的能力；不会改用其他模型或忽略需求。")
    if request.images and not capabilities.image_input:
        raise UnsupportedRecognitionCapability("所选视觉模型不支持图像输入。")
    if len(request.images) > 1 and not capabilities.multi_image_input:
        raise UnsupportedRecognitionCapability("所选视觉模型不支持多图输入；不会丢弃辅助图片。")


@dataclass(frozen=True)
class RecognitionResult:
    data: dict = field(repr=False)
    model_id: str
    model_version: str
    processing_time: float
    warnings: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not valid_identifier(self.model_id):
            raise ValueError("识别结果模型标识无效")
        if (not isinstance(self.model_version, str) or not self.model_version.strip() or len(self.model_version) > 128
                or any(ord(char) < 32 for char in self.model_version)):
            raise ValueError("识别结果模型版本无效")
        if (isinstance(self.processing_time, bool) or not isinstance(self.processing_time, (int, float))
                or not math.isfinite(self.processing_time) or self.processing_time < 0):
            raise ValueError("识别耗时必须为非负有限秒数")
        if (not isinstance(self.warnings, tuple) or len(self.warnings) > 100
                or any(not isinstance(item, str) or len(item) > 2000 for item in self.warnings)):
            raise ValueError("识别警告必须为有界文本元组")
        object.__setattr__(self, "data", bounded_json(self.data, max_bytes=2 * 1024 * 1024))
        metadata = bounded_json(self.metadata, max_bytes=16384)
        if any(not valid_identifier(key) or not isinstance(value, dict) for key, value in metadata.items()):
            raise ValueError("识别元数据需要模型命名空间")
        object.__setattr__(self, "metadata", metadata)


@runtime_checkable
class RecognitionBackend(Protocol):
    model_id: str
    model_version: str
    capabilities: RecognitionCapabilities

    def analyze(self, request: RecognitionRequest) -> RecognitionResult:
        """Analyze without changing input images or writing the object archive."""
