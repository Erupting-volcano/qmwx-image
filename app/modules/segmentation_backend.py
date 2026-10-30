"""Model-neutral requests/results; PIL data never belongs in profile JSON.

Every backend receives an EXIF-oriented RGB image. Prompt coordinates and all
returned masks refer to that same image. Backends own internal conversions;
the processor independently validates masks before transaction storage.
"""

from dataclasses import dataclass, field, fields
import math
import re
from typing import Literal, Protocol, runtime_checkable

from PIL import Image

from app.models.segmentation import validate_model_extensions


class SegmentationUnavailable(RuntimeError):
    """A safe, user-facing explanation for configuration or deployment failure."""


class SegmentationAPIError(RuntimeError):
    """A safe explanation for a failed remote segmentation request."""


class UnsupportedPromptError(SegmentationUnavailable):
    """The selected backend cannot process this prompt; never convert silently."""


class UnknownBackendError(SegmentationUnavailable):
    """The configured model has not been registered."""


class NotImplementedBackendError(SegmentationUnavailable):
    """The model is registered for future integration, not implemented."""


class BackendInitializationError(SegmentationUnavailable):
    """Sanitized backend initialization failure, without private configuration."""


def _number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class BoxPrompt:
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (not isinstance(self.box, tuple) or len(self.box) != 4 or not all(_number(x) for x in self.box)
                or not 0 <= self.box[0] < self.box[2] or not 0 <= self.box[1] < self.box[3]):
            raise ValueError("目标框必须为具有正面积的转正图像像素坐标")


@dataclass(frozen=True)
class PointPrompt:
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.points, tuple) or not self.points or len(self.points) > 1024
                or not isinstance(self.labels, tuple) or len(self.points) != len(self.labels)
                or any(not isinstance(p, tuple) or len(p) != 2 or not all(_number(x) and x >= 0 for x in p)
                       for p in self.points)
                or any(type(label) is not int or label not in {0, 1} for label in self.labels)):
            raise ValueError("点击提示需要对应的像素坐标和0/1背景前景标签")


@dataclass(frozen=True)
class MaskPrompt:
    mask: Image.Image

    def __post_init__(self) -> None:
        if not isinstance(self.mask, Image.Image) or self.mask.mode != "L" or min(self.mask.size) <= 0:
            raise ValueError("掩码提示必须为有效的L模式图像")
        if sum(self.mask.histogram()[1:255]):
            raise ValueError("掩码提示的像素必须为0或255")


@dataclass(frozen=True)
class TextPrompt:
    text: str

    def __post_init__(self) -> None:
        if (not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 2048
                or any(ord(char) < 32 for char in self.text)):
            raise ValueError("文字提示必须为1–2048个不含控制字符的有效字符")


@dataclass(frozen=True)
class AutomaticPrompt:
    """Explicit automatic masks; absence of a prompt is not automatic."""


SegmentationPrompt = BoxPrompt | PointPrompt | MaskPrompt | TextPrompt | AutomaticPrompt


@dataclass(frozen=True)
class SegmentationRequest:
    image: Image.Image
    prompt: SegmentationPrompt
    coordinate_system: Literal["exif_oriented_pixels"] = "exif_oriented_pixels"

    def __post_init__(self) -> None:
        if (not isinstance(self.image, Image.Image) or self.image.mode != "RGB"
                or min(self.image.size) <= 0 or self.coordinate_system != "exif_oriented_pixels"):
            raise ValueError("分割输入必须为已EXIF转正的RGB图像，坐标使用转正图像像素")
        prompt = self.prompt
        if not isinstance(prompt, (BoxPrompt, PointPrompt, MaskPrompt, TextPrompt, AutomaticPrompt)):
            raise UnsupportedPromptError("未识别的分割提示类型，不能忽略或自动转换提示。")
        # Frozen dataclasses still contain mutable PIL objects; recheck on use.
        if hasattr(prompt, "__post_init__"):
            prompt.__post_init__()
        width, height = self.image.size
        if isinstance(prompt, BoxPrompt) and (prompt.box[2] > width or prompt.box[3] > height):
            raise ValueError("目标框超出转正后图片范围")
        if isinstance(prompt, PointPrompt) and any(x >= width or y >= height for x, y in prompt.points):
            raise ValueError("点击坐标超出转正后图片范围")
        if isinstance(prompt, MaskPrompt) and prompt.mask.size != self.image.size:
            raise ValueError("提示掩码必须与转正后的输入图像同尺寸")


@dataclass(frozen=True)
class SegmentationCapabilities:
    box_prompt: bool = False
    point_prompt: bool = False
    mask_prompt: bool = False
    text_prompt: bool = False
    automatic_mask: bool = False
    multiple_masks: bool = False
    quality_score: bool = False
    iterative_refinement: bool = False

    def __post_init__(self) -> None:
        if any(type(getattr(self, item.name)) is not bool for item in fields(self)):
            raise ValueError("分割模型能力声明必须使用布尔值")


def check_request_capabilities(request: SegmentationRequest, capabilities: SegmentationCapabilities) -> None:
    if not isinstance(request, SegmentationRequest) or not isinstance(capabilities, SegmentationCapabilities):
        raise TypeError("必须提供标准分割请求和能力声明")
    request.__post_init__()
    supported = (
        (isinstance(request.prompt, BoxPrompt) and capabilities.box_prompt)
        or (isinstance(request.prompt, PointPrompt) and capabilities.point_prompt)
        or (isinstance(request.prompt, MaskPrompt) and capabilities.mask_prompt)
        or (isinstance(request.prompt, TextPrompt) and capabilities.text_prompt)
        or (isinstance(request.prompt, AutomaticPrompt) and capabilities.automatic_mask)
    )
    if not supported:
        raise UnsupportedPromptError("所选分割模型不支持此提示类型；请使用其声明的能力，不会自动转换提示。")


def _result_metadata(score_type: str | None) -> None:
    if score_type is not None and (not isinstance(score_type, str)
                                  or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", score_type)):
        raise ValueError("分数类型必须为清晰的短标识")


@dataclass(frozen=True)
class MaskCandidate:
    mask: Image.Image
    score: float | None = None
    score_type: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _result_metadata(self.score_type)


@dataclass(frozen=True)
class MaskPrediction:
    # Keep the original three positional arguments for injected adapters.
    mask: Image.Image
    score: float | None = None
    warnings: tuple[str, ...] = ()
    candidates: tuple[MaskCandidate, ...] = field(default=(), kw_only=True)
    score_type: str | None = field(default=None, kw_only=True)
    model_id: str | None = field(default=None, kw_only=True)
    model_version: str | None = field(default=None, kw_only=True)
    processing_time: float | None = field(default=None, kw_only=True)
    extensions: dict = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        _result_metadata(self.score_type)
        if not isinstance(self.candidates, tuple) or len(self.candidates) > 16 or any(
                not isinstance(candidate, MaskCandidate) for candidate in self.candidates):
            raise ValueError("候选掩码必须为最多16个标准候选结果")
        if self.model_id is not None and (not isinstance(self.model_id, str)
                                         or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", self.model_id)):
            raise ValueError("模型标识无效")
        if self.model_version is not None and (not isinstance(self.model_version, str)
                                              or not self.model_version.strip() or len(self.model_version) > 128
                                              or any(ord(c) < 32 for c in self.model_version)):
            raise ValueError("模型版本标识无效")
        if self.processing_time is not None and (not _number(self.processing_time) or self.processing_time < 0):
            raise ValueError("处理耗时必须为非负有限秒数")
        object.__setattr__(self, "extensions", validate_model_extensions(self.extensions))


@runtime_checkable
class SegmentationBackend(Protocol):
    model_id: str
    model_name: str
    model_version: str
    capabilities: SegmentationCapabilities

    def predict(self, request: SegmentationRequest) -> MaskPrediction:
        """Return owned L binary masks at request.image.size; never mutate input."""
