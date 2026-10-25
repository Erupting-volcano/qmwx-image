"""Standard module results and the typed image metadata payload."""

import json
from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue, field_validator, model_validator

from app.models._common import ContractModel
from app.models.image_asset import FORMAT_MIME_TYPES, ImageFormat, ImageMimeType

ProcessingStatus = Literal["success", "partial", "skipped", "failed", "needs_input"]


class ProcessingError(ContractModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)
    error_id: UUID | None = None
    asset_id: UUID | None = None


class ProcessingResult(ContractModel):
    module_name: str = Field(min_length=1, max_length=100)
    status: ProcessingStatus
    data: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ProcessingError] = Field(default_factory=list)

    @field_validator("data")
    @classmethod
    def ensure_standard_json(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        # JSON output must never contain JavaScript-only NaN or Infinity tokens.
        json.dumps(value, allow_nan=False)
        return value


class ImageMetadata(ContractModel):
    asset_id: UUID
    image_format: ImageFormat
    mime_type: ImageMimeType
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def match_format_and_mime(self) -> "ImageMetadata":
        if FORMAT_MIME_TYPES[self.image_format] != self.mime_type:
            raise ValueError("图片格式与 MIME 类型不一致")
        return self


class ImageMetadataData(ContractModel):
    images: list[ImageMetadata] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_images(self) -> "ImageMetadataData":
        ids = [image.asset_id for image in self.images]
        if len(ids) != len(set(ids)):
            raise ValueError("图片元数据中的资源标识不得重复")
        return self
