"""Image collection boundary for a parent project; independent of FastAPI."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import BinaryIO
from uuid import UUID, uuid4

from app.config import Settings
from app.models.image_asset import ImageAsset
from app.models.object_profile import ObjectProfile
from app.models.segmentation import NormalizedBox
from app.modules.base import BaseProcessor
from app.modules.image_metadata import ImageMetadataProcessor, InspectedImage, InvalidImageError
from app.modules.image_quality import ImageQualityProcessor
from app.modules.object_segmentation import ObjectSegmentationProcessor
from app.services.pipeline import Pipeline
from app.storage.local_storage import LocalStorage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageInput:
    """A filename for display and an open binary stream, owned by the caller."""

    filename: str
    stream: BinaryIO
    segmentation_box: NormalizedBox | None = None


class CollectionError(ValueError):
    """Expected invalid input; messages are safe for display by an adapter."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _safe_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(character for character in basename if ord(character) >= 32)
    return basename.strip()[:255] or "image"


class ImageCollectionService:
    """Collect and persist validated images with a versioned ObjectProfile output.

    A parent project may call this class without starting HTTP or using the demo UI.
    Streams are consumed from their current position and are never closed here.
    """

    def __init__(self, settings: Settings | None = None,
                 processors: Sequence[BaseProcessor] | None = None,
                 *, pipeline: Pipeline | None = None, storage: LocalStorage | None = None):
        if pipeline is not None and processors is not None:
            raise ValueError("Specify either processors or pipeline")
        self.settings = settings or Settings()
        self.storage = storage or LocalStorage(self.settings.data_dir)
        self.pipeline = pipeline or Pipeline(
            [ImageMetadataProcessor(), ImageQualityProcessor(), ObjectSegmentationProcessor(self.settings)]
            if processors is None else processors
        )

    def collect(self, name: str, images: Sequence[ImageInput]) -> ObjectProfile:
        """Validate the whole batch, run registered processors, atomically save it."""
        if not isinstance(name, str):
            raise CollectionError("invalid_name", "请填写物体名称。")
        name = name.strip()
        if not name or len(name) > 120 or any(ord(character) < 32 for character in name):
            raise CollectionError("invalid_name", "物体名称需要为 1–120 个字符，且不含控制字符。")
        if not images or len(images) > self.settings.max_files:
            raise CollectionError("invalid_images", f"请选择 1–{self.settings.max_files} 张图片。")
        if any(not isinstance(image, ImageInput) or not isinstance(image.filename, str)
               or not callable(getattr(image.stream, "read", None)) for image in images):
            raise CollectionError("invalid_images", "图片输入需要包含文件名和可读取的二进制流。")
        try:
            boxes = [NormalizedBox.model_validate(
                image.segmentation_box.model_dump() if isinstance(image.segmentation_box, NormalizedBox)
                else image.segmentation_box
            ) if image.segmentation_box is not None else None for image in images]
        except (ValueError, TypeError) as exc:
            raise CollectionError("invalid_segmentation_boxes", "目标框必须是转正后图片中有效的 0–1 比例坐标。") from exc

        object_id = uuid4()
        total_bytes = 0
        with self.storage.transaction(object_id) as transaction:
            assets: list[ImageAsset] = []
            inspected_images: dict[UUID, InspectedImage] = {}
            segmentation_boxes: dict[UUID, NormalizedBox] = {}
            for image, box in zip(images, boxes):
                asset_id = uuid4()
                filename = _safe_filename(image.filename)
                byte_budget = min(self.settings.max_image_bytes,
                                  self.settings.max_total_bytes - total_bytes)
                try:
                    temporary_image, size_bytes, checksum = transaction.stage_image(
                        image.stream, asset_id, byte_budget
                    )
                except OverflowError as exc:
                    # Preserve single-image error precedence when one read exceeds both limits.
                    if exc.args[0] > self.settings.max_image_bytes:
                        raise CollectionError("image_too_large", f"图片“{filename}”超过单张大小限制。", 413) from exc
                    raise CollectionError("upload_too_large", "本次图片总大小超过限制。", 413) from exc
                except TypeError as exc:
                    raise CollectionError("invalid_images", "图片流必须提供二进制数据。") from exc
                total_bytes += size_bytes
                try:
                    inspection = InspectedImage(temporary_image, max_pixels=self.settings.max_image_pixels)
                    metadata = inspection.metadata
                except InvalidImageError as exc:
                    raise CollectionError(exc.code, f"图片“{filename}”：{exc}", exc.status_code) from exc
                storage_path = transaction.finalize_image(temporary_image, metadata["image_format"])
                inspected_images[asset_id] = inspection
                if box is not None:
                    segmentation_boxes[asset_id] = box
                assets.append(ImageAsset(
                    asset_id=asset_id, filename=filename,
                    storage_path=storage_path,
                    mime_type=metadata["mime_type"], image_format=metadata["image_format"],
                    width=metadata["width"], height=metadata["height"],
                    size_bytes=size_bytes, checksum=checksum,
                ))

            initial = self.pipeline.assembler.create_profile(name, assets, object_id=object_id)
            profile = self.pipeline.run(initial, transaction.data_dir,
                                        max_image_pixels=self.settings.max_image_pixels,
                                        inspected_images=inspected_images,
                                        segmentation_boxes=segmentation_boxes, artifact_store=transaction)
            payload = transaction.save_profile(profile)
            transaction.commit()
        logger.info("Image collection %s saved: %s images, status=%s", object_id, len(assets), profile.status)
        # Return the exact contract representation which was persisted.
        return ObjectProfile.model_validate(payload)

    def get_profile(self, object_id: UUID | str) -> ObjectProfile:
        """Read a previously persisted collection, including after a process restart."""
        profile, _ = self.storage.read_profile(UUID(str(object_id)))
        return profile
