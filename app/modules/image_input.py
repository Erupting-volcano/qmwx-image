"""Shared safe, EXIF-oriented RGB input for visual processors."""

from PIL import Image, ImageOps

from app.models.image_asset import ImageAsset
from app.modules import image_metadata
from app.modules.base import ProcessingContext


def load_verified_rgb(context: ProcessingContext, asset: ImageAsset) -> Image.Image:
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
            raise image_metadata.InvalidImageError("视觉分析暂不支持带透明色的16位灰度图片，请转换为8位PNG或JPEG。")
        ImageOps.exif_transpose(image, in_place=True)
        if integer_grayscale:
            with image.point(lambda value: value * (255 / 65535)) as scaled:
                return scaled.convert("RGB")
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            with image.convert("RGBA") as rgba, Image.new("RGBA", image.size, "white") as background:
                background.alpha_composite(rgba)
                return background.convert("RGB")
        return image.convert("RGB")
