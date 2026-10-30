"""Typed box prompts and API segmentation records, including legacy local masks."""

import json
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from app.models._common import ContractModel, safe_relative_path


MODEL_NAME = "baidu_intelligent_segmentation"
SegmentationModel = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ScoreType = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
MAX_EXTENSION_BYTES = 16 * 1024
PixelCoordinate = Annotated[int, Field(ge=0, strict=True)]
SegmentationStatus = Literal[
    "success", "needs_confirmation", "not_found", "failed", "disabled", "unavailable", "needs_input",
]


def validate_model_extensions(value: object) -> dict:
    """Small, namespaced JSON only; masks/logits/state binaries belong in files."""
    if not isinstance(value, dict) or len(value) > 16:
        raise ValueError("模型扩展必须是有命名空间的小型JSON对象")

    def check(item: object, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError("模型扩展层级过深")
        if item is None or type(item) in {str, int, float, bool}:
            return
        if isinstance(item, list):
            if len(item) > 256:
                raise ValueError("模型扩展列表过长")
            for child in item:
                check(child, depth + 1)
        elif isinstance(item, dict) and all(isinstance(key, str) for key in item):
            if len(item) > 256:
                raise ValueError("模型扩展对象过大")
            for child in item.values():
                check(child, depth + 1)
        else:
            raise ValueError("模型扩展仅允许JSON类型，不接受二进制或模型对象")

    for namespace, payload in value.items():
        if (not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", namespace)
                or not isinstance(payload, dict)):
            raise ValueError("模型扩展必须使用有效模型标识作为命名空间")
        check(payload)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(content.encode("utf-8")) > MAX_EXTENSION_BYTES:
        raise ValueError("模型扩展超过16KiB，较大数据应使用独立文件")
    return json.loads(content)


class SegmentationCandidate(ContractModel):
    """An additional mask using the same source size and coordinates as its parent."""

    mask_path: str
    bbox: tuple[PixelCoordinate, PixelCoordinate, PixelCoordinate, PixelCoordinate]
    area_ratio: float = Field(gt=0, le=1)
    score: float | None = None
    score_type: ScoreType | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("mask_path")
    @classmethod
    def safe_mask_path(cls, value: str) -> str:
        value = safe_relative_path(value)
        if not value.endswith(".png"):
            raise ValueError("候选掩码必须保存为PNG")
        return value

    @model_validator(mode="after")
    def consistent_score(self) -> "SegmentationCandidate":
        if self.score is None and self.score_type is not None:
            raise ValueError("无分数时不得声明分数类型")
        return self


class NormalizedBox(ContractModel):
    """Box edges in the EXIF-oriented display, normalized to [0, 1]."""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True, frozen=True, revalidate_instances="always",
    )
    x0: float = Field(ge=0, le=1, strict=True)
    y0: float = Field(ge=0, le=1, strict=True)
    x1: float = Field(ge=0, le=1, strict=True)
    y1: float = Field(ge=0, le=1, strict=True)

    @model_validator(mode="after")
    def ordered_edges(self) -> "NormalizedBox":
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("目标框必须具有正宽度和高度")
        return self

    def to_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸必须大于零")
        return self.x0 * width, self.y0 * height, self.x1 * width, self.y1 * height


class SegmentationImage(ContractModel):
    asset_id: UUID
    filename: str = Field(min_length=1, max_length=512)
    status: SegmentationStatus
    model_name: SegmentationModel = MODEL_NAME
    model_id: SegmentationModel | None = None
    model_version: str | None = Field(default=None, min_length=1, max_length=128)
    score: float | None = None
    score_type: ScoreType | None = None
    processing_time: float | None = Field(default=None, ge=0)
    extensions: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    candidates: list[SegmentationCandidate] = Field(default_factory=list, max_length=16)
    prompt_type: Literal["box", "none"] = "none"
    prompt_box: NormalizedBox | None = None
    width: int | None = Field(default=None, gt=0, strict=True)
    height: int | None = Field(default=None, gt=0, strict=True)
    coordinate_system: Literal["exif_oriented_pixels"] = "exif_oriented_pixels"
    bbox: tuple[PixelCoordinate, PixelCoordinate, PixelCoordinate, PixelCoordinate] | None = Field(
        default=None, description="[x0,y0,x1,y1], top-left origin, right/bottom edges exclusive."
    )
    mask_path: str | None = None
    area_ratio: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(
        default=None, ge=0, le=1,
        description="Provider quality score when supplied; not target identity probability. Baidu returns no verified score.",
    )
    warnings: list[str] = Field(default_factory=list)
    error_id: UUID | None = None

    @field_validator("extensions", mode="before")
    @classmethod
    def small_namespaced_extensions(cls, value: object) -> dict:
        return validate_model_extensions(value)

    def stored_masks(self) -> tuple:
        """Primary and candidate file records share existing safety checks."""
        return ((self,) if self.mask_path is not None else ()) + tuple(self.candidates)

    @field_validator("mask_path")
    @classmethod
    def safe_mask_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = safe_relative_path(value)
        if not value.endswith(".png"):
            raise ValueError("分割掩码必须保存为PNG")
        return value

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("文件名不得包含控制字符")
        return value

    @model_validator(mode="after")
    def consistent_result(self) -> "SegmentationImage":
        if (self.width is None) != (self.height is None):
            raise ValueError("分割图像宽高必须同时存在或同时为空")
        if (self.prompt_type == "box") != (self.prompt_box is not None):
            raise ValueError("提示类型与目标框不一致")
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if (self.width is None or self.height is None or min(x0, y0) < 0
                    or x0 >= x1 or y0 >= y1 or x1 > self.width or y1 > self.height):
                raise ValueError("分割边界框超出图像范围或为空")
        has_mask = self.status in {"success", "needs_confirmation"}
        if has_mask:
            if (self.mask_path is None or self.bbox is None or self.width is None
                    or self.area_ratio is None or self.area_ratio <= 0 or self.prompt_box is None):
                raise ValueError("有效分割结果必须具有掩码、尺寸、边界框、面积和提示框")
            if self.error_id is not None:
                raise ValueError("有效掩码结果不得同时声明执行错误")
        elif self.mask_path is not None or self.bbox is not None:
            raise ValueError("未生成有效掩码的状态不得声明掩码路径或边界框")
        if self.status in {"disabled", "unavailable", "needs_input", "failed"} and (
                self.area_ratio is not None or self.confidence is not None):
            raise ValueError("未执行或失败结果不得声明分割面积或质量分数")
        if self.status == "not_found" and self.area_ratio != 0:
            raise ValueError("未找到前景时面积必须为零")
        if self.status == "success" and self.area_ratio == 1:
            raise ValueError("全图掩码需要人工确认")
        if self.status in {"failed", "not_found", "unavailable"} and self.error_id is None:
            raise ValueError("失败或不可用结果必须提供安全错误编号")
        if self.score is None and self.confidence is None and self.score_type is not None:
            raise ValueError("无分数时不得声明分数类型")
        if self.status in {"disabled", "unavailable", "needs_input", "failed", "not_found"} and self.score is not None:
            raise ValueError("未生成掩码时不得声明预测分数")
        if self.candidates and not has_mask:
            raise ValueError("没有主要掩码时不得声明候选掩码")
        paths = [record.mask_path for record in self.stored_masks()]
        if len(paths) != len(set(paths)):
            raise ValueError("主要掩码与候选掩码不得重复引用文件")
        for candidate in self.candidates:
            x0, y0, x1, y1 = candidate.bbox
            if (self.width is None or self.height is None or x0 >= x1 or y0 >= y1
                    or x1 > self.width or y1 > self.height):
                raise ValueError("候选掩码边界框超出图像范围或为空")
        return self


class SegmentationData(ContractModel):
    input_count: int = Field(ge=0, strict=True)
    success_count: int = Field(ge=0, strict=True)
    review_count: int = Field(ge=0, strict=True)
    failed_count: int = Field(ge=0, strict=True)
    skipped_count: int = Field(ge=0, strict=True)
    images: list[SegmentationImage] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_counts(self) -> "SegmentationData":
        if self.input_count != len(self.images):
            raise ValueError("分割输入数量必须与逐图状态数量一致")
        expected = (
            sum(image.status == "success" for image in self.images),
            sum(image.status == "needs_confirmation" for image in self.images),
            sum(image.status in {"failed", "not_found"} for image in self.images),
            sum(image.status in {"disabled", "unavailable", "needs_input"} for image in self.images),
        )
        if expected != (self.success_count, self.review_count, self.failed_count, self.skipped_count):
            raise ValueError("分割汇总数量与逐图状态不一致")
        identifiers = [image.asset_id for image in self.images]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("分割结果中的图片标识不得重复")
        return self
