"""Box-guided SAM 2 masks, using the caller's image validation and transaction."""

import logging
import math
from uuid import uuid4

from PIL import Image, ImageOps

from app.config import Settings
from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQualityData
from app.models.segmentation import NormalizedBox, SegmentationData, SegmentationImage
from app.models.processing_result import ProcessingError, ProcessingResult
from app.modules import image_metadata
from app.modules.base import BaseProcessor, ProcessingContext
from app.modules.sam2_backend import MaskPrediction, Sam2Backend, SegmentationUnavailable


logger = logging.getLogger(__name__)
MIN_REVIEW_AREA_RATIO = 0.001
MAX_REVIEW_AREA_RATIO = 0.98
MIN_REVIEW_MASK_SCORE = 0.7
TARGET_REVIEW_NOTE = "请人工核对掩码是否覆盖所选目标；模型质量分数不是目标识别正确率。"


class ObjectSegmentationProcessor(BaseProcessor):
    name = "object_segmentation"

    def __init__(self, settings: Settings | None = None, *, backend=None) -> None:
        self.settings = settings or Settings()
        self.backend = backend if backend is not None else Sam2Backend(
            self.settings.segmentation_checkpoint, self.settings.segmentation_device,
        )

    @staticmethod
    def _load_rgb(context: ProcessingContext, asset: ImageAsset) -> Image.Image:
        path = context.resolve_asset(asset)
        inspection = context.inspected_images.get(asset.asset_id)
        metadata = (
            inspection.metadata_if_unchanged(path, context.max_image_pixels)
            if type(inspection) is image_metadata.InspectedImage else None
        )
        if metadata is None:
            metadata = image_metadata.inspect_image(path, context.max_image_pixels)
        with Image.open(path) as image:
            if image.format != metadata["image_format"] or image.size != (metadata["width"], metadata["height"]):
                raise image_metadata.InvalidImageError("图片内容在验证后发生变化")
            integer_grayscale = image.mode in {"I;16", "I;16L", "I;16B", "I"}
            if integer_grayscale and "transparency" in image.info:
                raise image_metadata.InvalidImageError("分割暂不支持带透明色的16位灰度图片，请转换为8位PNG或JPEG。")
            ImageOps.exif_transpose(image, in_place=True)
            if integer_grayscale:
                with image.point(lambda value: value * (255 / 65535)) as scaled:
                    return scaled.convert("RGB")
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                with image.convert("RGBA") as rgba, Image.new("RGBA", image.size, "white") as background:
                    background.alpha_composite(rgba)
                    return background.convert("RGB")
            return image.convert("RGB")

    def _segment_asset(self, context: ProcessingContext, asset: ImageAsset) -> SegmentationImage:
        prompt = context.segmentation_boxes.get(asset.asset_id)
        if not self.settings.segmentation_enabled:
            return SegmentationImage(
                asset_id=asset.asset_id, filename=asset.filename, status="disabled",
                warnings=["物体分割未启用；现有图片上传与质量评估仍可使用。"],
            )
        if prompt is None:
            return SegmentationImage(
                asset_id=asset.asset_id, filename=asset.filename, status="needs_input",
                warnings=["请在上传照片上框选一个目标；本版本不会根据物体名称自动定位。"],
            )
        prompt = NormalizedBox.model_validate(prompt.model_dump() if isinstance(prompt, NormalizedBox) else prompt)
        if context.artifact_store is None:
            raise SegmentationUnavailable("本次处理没有提供事务产物存储，无法安全保存分割掩码。")
        with self._load_rgb(context, asset) as rgb:
            width, height = rgb.size
            prediction = self.backend.predict(rgb, prompt.to_pixels(width, height))
        if not isinstance(prediction, MaskPrediction) or not isinstance(prediction.mask, Image.Image):
            raise ValueError("Backend must return a real mask image")
        with prediction.mask as mask:
            if mask.mode != "L" or mask.size != (width, height):
                raise ValueError("Mask must be L at the oriented source dimensions")
            histogram = mask.histogram()
            if sum(histogram[1:255]):
                raise ValueError("Mask pixels must be 0 or 255")
            foreground = histogram[255]
            area_ratio = foreground / (width * height)
            bbox = mask.getbbox()
            if foreground == 0 or bbox is None:
                return SegmentationImage(
                    asset_id=asset.asset_id, filename=asset.filename, status="not_found",
                    prompt_type="box", prompt_box=prompt, width=width, height=height,
                    area_ratio=0, warnings=["模型没有在框选范围中生成前景，请调整目标框或补拍照片。"],
                    error_id=uuid4(),
                )
            warnings = [TARGET_REVIEW_NOTE]
            for previous in reversed(context.previous_results):
                if previous.module_name != "image_quality" or previous.status not in {"success", "partial"}:
                    continue
                try:
                    quality = ImageQualityData.model_validate(previous.data)
                    if any(item.asset_id == asset.asset_id and item.status == "review" for item in quality.images):
                        warnings.append("该图片的质量筛查建议复查，曝光、对比度或细节不足可能影响分割边界。")
                except ValueError:
                    pass
                break
            needs_review = area_ratio <= MIN_REVIEW_AREA_RATIO or area_ratio >= MAX_REVIEW_AREA_RATIO
            if needs_review:
                warnings.append("掩码面积过小或几乎覆盖整张图片，必须人工复查并调整目标框。")
            score = prediction.score
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                score = None
                needs_review = True
                warnings.append("模型未提供有效的掩码质量分数，请人工确认。")
            elif score < MIN_REVIEW_MASK_SCORE:
                needs_review = True
                warnings.append("模型给出的掩码质量分数偏低，请人工确认边界。")
            mask_path = context.artifact_store.save_mask(asset.asset_id, mask)
        return SegmentationImage(
            asset_id=asset.asset_id, filename=asset.filename,
            status="needs_confirmation" if needs_review else "success",
            prompt_type="box", prompt_box=prompt, width=width, height=height,
            mask_path=mask_path, bbox=bbox, area_ratio=area_ratio, confidence=score, warnings=warnings,
        )

    def process(self, context: ProcessingContext) -> ProcessingResult:
        images, errors, warnings = [], [], []
        for asset in context.images:
            try:
                item = self._segment_asset(context, asset)
            except Exception as exc:
                error_id = uuid4()
                logger.exception("物体分割失败 error_id=%s asset_id=%s", error_id, asset.asset_id)
                unavailable = isinstance(exc, SegmentationUnavailable)
                message = (str(exc) if unavailable or isinstance(exc, image_metadata.InvalidImageError)
                           else "此图片分割失败，原始图片仍已保留，请根据错误编号检查日志或重新框选。")
                raw_prompt = context.segmentation_boxes.get(asset.asset_id)
                try:
                    prompt = NormalizedBox.model_validate(
                        raw_prompt.model_dump() if isinstance(raw_prompt, NormalizedBox) else raw_prompt
                    ) if raw_prompt is not None else None
                except (ValueError, TypeError):
                    prompt = None
                item = SegmentationImage(
                    asset_id=asset.asset_id, filename=asset.filename,
                    status="unavailable" if unavailable else "failed",
                    prompt_type="box" if prompt is not None else "none", prompt_box=prompt,
                    warnings=[message], error_id=error_id,
                )
            images.append(item)
            warnings.extend(f"图片“{item.filename}”：{warning}" for warning in item.warnings)
            if item.error_id is not None:
                errors.append(ProcessingError(
                    code=f"segmentation_{item.status}", message=item.warnings[0],
                    error_id=item.error_id, asset_id=item.asset_id,
                ))
        payload = SegmentationData(
            input_count=len(images), success_count=sum(item.status == "success" for item in images),
            review_count=sum(item.status == "needs_confirmation" for item in images),
            failed_count=sum(item.status in {"failed", "not_found"} for item in images),
            skipped_count=sum(item.status in {"disabled", "unavailable", "needs_input"} for item in images),
            images=images,
        )
        if payload.success_count or payload.review_count:
            status = "success" if payload.success_count == payload.input_count else "partial"
        elif payload.failed_count:
            status = "failed"
        elif any(item.status == "needs_input" for item in images):
            status = "needs_input"
        else:
            status = "skipped"
        return ProcessingResult(
            module_name=self.name, status=status, data=payload.model_dump(mode="json"),
            warnings=warnings, errors=errors,
        )
