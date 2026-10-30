"""Small backend contract shared by the segmentation processor and API adapter."""

from dataclasses import dataclass

from PIL import Image


class SegmentationUnavailable(RuntimeError):
    """A safe explanation for missing API configuration or authentication."""


class SegmentationAPIError(RuntimeError):
    """A safe explanation for a failed remote segmentation request."""


@dataclass(frozen=True)
class MaskPrediction:
    mask: Image.Image
    score: float | None = None
    warnings: tuple[str, ...] = ()
