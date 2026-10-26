"""Actual static image decoding and a minimal working processing module."""

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.models.image_asset import FORMAT_MIME_TYPES
from app.models.processing_result import ImageMetadata, ImageMetadataData, ProcessingError, ProcessingResult
from app.modules.base import BaseProcessor, ProcessingContext

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """Image rejected with a user-safe, Chinese error message."""

    def __init__(self, message: str, code: str = "invalid_image", status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def inspect_image(path: Path, max_pixels: int = 40_000_000) -> dict:
    """Verify file structure, reopen, and decode pixels before accepting an image."""
    if max_pixels <= 0:
        raise InvalidImageError("图片像素限制必须大于零")
    try:
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise InvalidImageError("图片文件为空")
        with Image.open(path) as image:
            image_format = image.format
            if image_format not in FORMAT_MIME_TYPES:
                raise InvalidImageError("仅支持 JPEG、PNG 和 WebP 格式图片", "unsupported_image", 415)
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise InvalidImageError("图片像素数超出安全限制", "image_too_large", 413)
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                raise InvalidImageError("当前版本仅支持静态图片，请上传单帧图片", "unsupported_image", 415)
            image.verify()
        with Image.open(path) as image:
            if image.format != image_format or image.size != (width, height):
                raise InvalidImageError("图片内容不一致，请重新选择文件")
            image.load()
        return {
            "image_format": image_format,
            "mime_type": FORMAT_MIME_TYPES[image_format],
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
        }
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("图片像素数超出安全限制", "image_too_large", 413) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise InvalidImageError("图片无法完整解码，文件可能损坏或格式不受支持") from exc


@dataclass(frozen=True, init=False, slots=True)
class InspectedImage:
    """A call-local receipt, created only by fully inspecting a real file.

    The same-stem rule permits the transaction's .upload-to-format rename, while
    the fingerprint rejects a replaced or modified file. No receipt is persisted.
    """

    _stem_path: Path
    _fingerprint: tuple[int, int, int, int]
    _metadata_items: tuple[tuple[str, str | int], ...]
    _max_pixels: int

    def __init__(self, path: Path, max_pixels: int = 40_000_000) -> None:
        try:
            source = Path(path).resolve()
            before = self._file_fingerprint(source)
            metadata = inspect_image(source, max_pixels)
            after = self._file_fingerprint(source)
        except OSError as exc:
            raise InvalidImageError("图片无法读取，请重新选择文件") from exc
        if before != after:
            raise InvalidImageError("图片在验证期间发生变化，请重新选择文件")
        object.__setattr__(self, "_stem_path", source.with_suffix(""))
        object.__setattr__(self, "_fingerprint", after)
        object.__setattr__(self, "_metadata_items", tuple(metadata.items()))
        object.__setattr__(self, "_max_pixels", max_pixels)

    @staticmethod
    def _file_fingerprint(path: Path) -> tuple[int, int, int, int]:
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    @property
    def metadata(self) -> dict:
        """Return a copy of the metadata produced by the completed inspection."""
        return dict(self._metadata_items)

    def metadata_if_unchanged(self, path: Path, max_pixels: int = 40_000_000) -> dict | None:
        """Return a fresh metadata dict only for the same file and safety limit."""
        if max_pixels < self._max_pixels or max_pixels <= 0:
            return None
        try:
            source = Path(path).resolve()
            if source.with_suffix("") != self._stem_path:
                return None
            if self._file_fingerprint(source) != self._fingerprint:
                return None
        except (OSError, ValueError):
            return None
        return self.metadata


class ImageMetadataProcessor(BaseProcessor):
    name = "image_metadata"

    def process(self, context: ProcessingContext) -> ProcessingResult:
        images: list[ImageMetadata] = []
        errors: list[ProcessingError] = []
        for asset in context.images:
            try:
                path = context.resolve_asset(asset)
                inspection = context.inspected_images.get(asset.asset_id)
                metadata = (
                    inspection.metadata_if_unchanged(path, context.max_image_pixels)
                    if type(inspection) is InspectedImage else None
                )
                if metadata is None:
                    metadata = inspect_image(path, context.max_image_pixels)
                images.append(ImageMetadata(asset_id=asset.asset_id, **metadata))
            except Exception:
                error_id = uuid4()
                logger.exception("图片信息提取失败 error_id=%s asset_id=%s", error_id, asset.asset_id)
                errors.append(ProcessingError(
                    code="image_metadata_failed",
                    message="无法读取此图片的基础信息，请根据错误编号检查服务日志。",
                    error_id=error_id,
                    asset_id=asset.asset_id,
                ))
        status = "partial" if images and errors else "failed" if errors else "success" if images else "skipped"
        return ProcessingResult(
            module_name=self.name,
            status=status,
            data=ImageMetadataData(images=images).model_dump(mode="json"),
            warnings=["没有可供处理的图片。"] if not context.images else [],
            errors=errors,
        )
