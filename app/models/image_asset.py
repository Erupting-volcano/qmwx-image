"""An uploaded image stored below the configured data directory."""

from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from app.models._common import ContractModel, safe_relative_path

ImageFormat = Literal["JPEG", "PNG", "WEBP"]
ImageMimeType = Literal["image/jpeg", "image/png", "image/webp"]
FORMAT_MIME_TYPES: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageAsset(ContractModel):
    asset_id: UUID = Field(default_factory=uuid4)
    filename: str = Field(min_length=1, max_length=512)
    storage_path: str
    mime_type: ImageMimeType
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    checksum: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    image_format: ImageFormat | None = None

    _validate_storage_path = field_validator("storage_path")(safe_relative_path)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("文件名不得包含控制字符")
        return value

    @field_validator("checksum")
    @classmethod
    def lowercase_checksum(cls, value: str | None) -> str | None:
        return value.lower() if value else None

    @model_validator(mode="after")
    def match_format_and_mime(self) -> "ImageAsset":
        if self.image_format and FORMAT_MIME_TYPES[self.image_format] != self.mime_type:
            raise ValueError("图片格式与 MIME 类型不一致")
        return self
