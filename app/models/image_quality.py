"""Validated, per-image heuristic quality results; no physical dimensions implied."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.models._common import ContractModel


QualityIssue = Literal["low_resolution", "too_dark", "too_bright", "low_contrast", "low_detail"]
Suggestion = Annotated[str, Field(min_length=1, max_length=500)]
ASSESSMENT_NOTE = (
    "仅为启发式质量筛查；低细节响应也可能来自平滑表面，不能据此诊断模糊。"
    "good 不保证照片适合三维重建，也不表示尺寸信息可信。"
)


class ImageQuality(ContractModel):
    asset_id: UUID
    filename: str = Field(min_length=1, max_length=512)
    width: int = Field(gt=0, strict=True, description="Original encoded pixel width, before EXIF rotation.")
    height: int = Field(gt=0, strict=True, description="Original encoded pixel height, before EXIF rotation.")
    analysis_width: int = Field(gt=0, le=2048, strict=True)
    analysis_height: int = Field(gt=0, le=2048, strict=True)
    mean_luminance: float = Field(ge=0, le=255)
    contrast_stddev: float = Field(ge=0, le=127.5)
    edge_strength: float = Field(ge=0, le=255)
    status: Literal["good", "review"]
    issues: list[QualityIssue] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)

    @field_validator("filename")
    @classmethod
    def safe_filename_text(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("文件名不得包含控制字符")
        return value

    @model_validator(mode="after")
    def consistent_findings(self) -> "ImageQuality":
        if len(self.issues) != len(set(self.issues)):
            raise ValueError("同一图片的问题代码不得重复")
        if len(self.issues) != len(self.suggestions):
            raise ValueError("每个问题必须对应一条建议")
        if (self.status == "review") != bool(self.issues):
            raise ValueError("复查状态必须与问题列表一致")
        # EXIF rotation may swap the two axes, but analysis never upsizes images.
        if (max(self.analysis_width, self.analysis_height) > max(self.width, self.height)
                or min(self.analysis_width, self.analysis_height) > min(self.width, self.height)):
            raise ValueError("分析尺寸不得大于原图尺寸")
        return self


class ImageQualityData(ContractModel):
    input_count: int = Field(ge=0, strict=True)
    checked_count: int = Field(ge=0, strict=True)
    review_count: int = Field(ge=0, strict=True)
    images: list[ImageQuality] = Field(default_factory=list)
    method: Literal["pillow_luminance_gradient_v1"] = "pillow_luminance_gradient_v1"
    assessment_note: Literal[ASSESSMENT_NOTE] = ASSESSMENT_NOTE

    @model_validator(mode="after")
    def consistent_counts(self) -> "ImageQualityData":
        if self.checked_count != len(self.images) or self.checked_count > self.input_count:
            raise ValueError("检查数量必须与图片结果一致，且不能超过输入数量")
        if self.review_count != sum(image.status == "review" for image in self.images):
            raise ValueError("复查数量必须与图片结果一致")
        identifiers = [image.asset_id for image in self.images]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("质量结果的图片标识不得重复")
        return self
