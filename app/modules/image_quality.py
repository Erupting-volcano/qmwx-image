"""Small Pillow quality heuristics, without a blur diagnosis or physical claims."""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageChops, ImageOps, ImageStat

from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQuality, ImageQualityData
from app.models.processing_result import ProcessingError, ProcessingResult
from app.modules.base import BaseProcessor, ProcessingContext
from app.modules import image_metadata


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityThresholds:
    """Initial screening thresholds, to be calibrated on real capture examples."""

    min_short_side: int = 480
    min_pixels: int = 640 * 480
    max_analysis_side: int = 1024
    dark_mean: float = 45.0
    bright_mean: float = 210.0
    low_contrast_stddev: float = 20.0
    low_edge_strength: float = 4.0

    def __post_init__(self) -> None:
        for name in ("min_short_side", "min_pixels", "max_analysis_side"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.max_analysis_side <= 2048:
            raise ValueError("max_analysis_side must be between 2 and 2048")
        for name in ("dark_mean", "bright_mean", "low_contrast_stddev", "low_edge_strength"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.dark_mean < self.bright_mean <= 255:
            raise ValueError("Brightness thresholds must satisfy 0 <= dark < bright <= 255")
        if not 0 <= self.low_contrast_stddev <= 127.5 or not 0 <= self.low_edge_strength <= 255:
            raise ValueError("Contrast/edge thresholds are outside their metric ranges")


DEFAULT_THRESHOLDS = QualityThresholds()


def _load_luminance(path: Path, metadata: dict, max_side: int) -> Image.Image:
    """Decode the verified file; only create derivative images after shrinking.

    JPEG thumbnail decoding may use draft reduction. PNG/WebP can still require
    a full original decode, bounded by the existing max_image_pixels policy.
    """
    with Image.open(path) as image:
        if image.format != metadata["image_format"] or image.size != (metadata["width"], metadata["height"]):
            raise image_metadata.InvalidImageError("图片内容在验证后发生变化")
        integer_grayscale = image.mode in {"I;16", "I;16L", "I;16B", "I"}
        if integer_grayscale and "transparency" in image.info:
            # Pillow's RGBA conversion clips 16-bit values. A reliable alpha mask
            # would need original-resolution work before thumbnail interpolation.
            raise image_metadata.InvalidImageError(
                "质量评估暂不支持带透明色的16位灰度图片，请转换为8位PNG或JPEG后重试。"
            )
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        ImageOps.exif_transpose(image, in_place=True)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            with image.convert("RGBA") as rgba, Image.new("RGBA", image.size, "white") as background:
                background.alpha_composite(rgba)
                return background.convert("L")
        if integer_grayscale:
            # PNG integer grayscale is 16-bit; a direct L conversion would clip.
            with image.point(lambda value: value * (255 / 65535)) as scaled:
                return scaled.convert("L")
        return image.convert("L")


def _edge_strength(gray: Image.Image) -> float:
    """Mean absolute difference of horizontal/vertical neighbours, in 0..255."""
    width, height = gray.size
    weighted_sum = 0.0
    pair_count = 0
    boxes = []
    if width > 1:
        boxes.append(((0, 0, width - 1, height), (1, 0, width, height)))
    if height > 1:
        boxes.append(((0, 0, width, height - 1), (0, 1, width, height)))
    for first_box, second_box in boxes:
        with gray.crop(first_box) as first, gray.crop(second_box) as second:
            with ImageChops.difference(first, second) as difference:
                count = difference.width * difference.height
                weighted_sum += ImageStat.Stat(difference).mean[0] * count
                pair_count += count
    return weighted_sum / pair_count if pair_count else 0.0


class ImageQualityProcessor(BaseProcessor):
    name = "image_quality"

    def __init__(self, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> None:
        if not isinstance(thresholds, QualityThresholds):
            raise TypeError("thresholds must be QualityThresholds")
        self.thresholds = thresholds

    def _assess_asset(self, context: ProcessingContext, asset: ImageAsset) -> ImageQuality:
        path = context.resolve_asset(asset)
        inspection = context.inspected_images.get(asset.asset_id)
        metadata = (
            inspection.metadata_if_unchanged(path, context.max_image_pixels)
            if type(inspection) is image_metadata.InspectedImage else None
        )
        if metadata is None:
            metadata = image_metadata.inspect_image(path, context.max_image_pixels)
        with _load_luminance(path, metadata, self.thresholds.max_analysis_side) as gray:
            statistics = ImageStat.Stat(gray)
            mean_luminance = statistics.mean[0]
            contrast = min(127.5, statistics.stddev[0])
            edge = _edge_strength(gray)
            analysis_width, analysis_height = gray.size

        width, height = metadata["width"], metadata["height"]
        findings = []
        if min(width, height) < self.thresholds.min_short_side or width * height < self.thresholds.min_pixels:
            findings.append(("low_resolution", "图片分辨率偏低，请使用较高分辨率拍摄，并让物体占据更多画面。"))
        if mean_luminance < self.thresholds.dark_mean:
            findings.append(("too_dark", "画面整体偏暗，请增加均匀照明或适当提高曝光，避免物体细节落入阴影。"))
        if mean_luminance > self.thresholds.bright_mean:
            findings.append(("too_bright", "画面整体偏亮，请适当降低曝光并避免强光直射或反光，检查高光处细节。"))
        if contrast < self.thresholds.low_contrast_stddev:
            findings.append(("low_contrast", "画面整体对比度偏低，请选择与物体有明显区分的背景，并改善照明。"))
        if edge < self.thresholds.low_edge_strength:
            findings.append(("low_detail", "边缘细节响应较弱，可能来自平滑表面或失焦；请人工查看物体细节，必要时对焦并稳住手机重拍。"))
        return ImageQuality(
            asset_id=asset.asset_id, filename=asset.filename, width=width, height=height,
            analysis_width=analysis_width, analysis_height=analysis_height,
            mean_luminance=mean_luminance, contrast_stddev=contrast, edge_strength=edge,
            status="review" if findings else "good",
            issues=[code for code, _ in findings], suggestions=[suggestion for _, suggestion in findings],
        )

    def process(self, context: ProcessingContext) -> ProcessingResult:
        images = []
        errors = []
        warnings = []
        for asset in context.images:
            try:
                result = self._assess_asset(context, asset)
                images.append(result)
                if result.status == "review":
                    warnings.append(f"图片“{result.filename}”需要复查：" + "；".join(result.suggestions))
            except Exception as exc:
                error_id = uuid4()
                logger.exception("图片质量评估失败 error_id=%s asset_id=%s", error_id, asset.asset_id)
                errors.append(ProcessingError(
                    code="image_quality_failed",
                    message=(str(exc) if isinstance(exc, image_metadata.InvalidImageError)
                             else "此图片的质量评估未完成，请根据错误编号检查服务日志或重新拍摄。"),
                    error_id=error_id, asset_id=asset.asset_id,
                ))
        payload = ImageQualityData(
            input_count=len(context.images), checked_count=len(images),
            review_count=sum(image.status == "review" for image in images), images=images,
        )
        status = "partial" if images and errors else "failed" if errors else "success" if images else "skipped"
        return ProcessingResult(
            module_name=self.name, status=status, data=payload.model_dump(mode="json"),
            warnings=warnings if context.images else ["没有可供质量评估的图片。"], errors=errors,
        )
