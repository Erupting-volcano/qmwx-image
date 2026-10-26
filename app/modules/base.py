"""Stable module interface, independent of HTTP and persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import UUID

from app.models.image_asset import ImageAsset
from app.models.object_profile import ObjectProfile
from app.models.processing_result import ProcessingResult

if TYPE_CHECKING:
    from app.modules.image_metadata import InspectedImage


@dataclass(frozen=True)
class ProcessingContext:
    images: tuple[ImageAsset, ...]
    data_dir: Path
    profile: ObjectProfile | None = None
    previous_results: tuple[ProcessingResult, ...] = ()
    max_image_pixels: int = 40_000_000
    inspected_images: Mapping[UUID, InspectedImage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Each module gets a read-only copy; the receipts themselves are immutable.
        object.__setattr__(self, "inspected_images", MappingProxyType(dict(self.inspected_images)))

    def resolve_asset(self, asset: ImageAsset) -> Path:
        """Resolve below the data root, also rejecting symlinks escaping that root."""
        validated = ImageAsset.model_validate(asset.model_dump())
        registered = next((image for image in self.images if image.asset_id == validated.asset_id), None)
        if registered is None or registered.storage_path != validated.storage_path:
            raise ValueError("图片资源不属于当前处理上下文")
        root = self.data_dir.resolve()
        candidate = (root / validated.storage_path).resolve()
        if not candidate.is_relative_to(root) or candidate == root:
            raise ValueError("图片资源路径超出数据目录")
        return candidate


class BaseProcessor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique stable module name used to register a merge adapter."""

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Return a result; never write or overwrite the object profile."""
