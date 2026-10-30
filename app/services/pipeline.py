"""Sequential execution with isolated processor snapshots and failure boundaries."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from app.models.object_profile import ObjectProfile
from app.models.processing_result import ProcessingError, ProcessingResult
from app.modules.base import BaseProcessor, ProcessingContext
from app.services.profile_assembler import ProfileAssembler
from app.storage.local_storage import resolve_mask_path

if TYPE_CHECKING:
    from app.models.segmentation import NormalizedBox
    from app.modules.image_metadata import InspectedImage
    from app.storage.local_storage import ObjectTransaction

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self, processors: Sequence[BaseProcessor], assembler: ProfileAssembler | None = None,
    ) -> None:
        self.processors = tuple(processors)
        names = [processor.name for processor in self.processors]
        if any(not isinstance(name, str) or not name.strip() or len(name) > 100 for name in names):
            raise ValueError("每个模块必须具有有效名称")
        self.assembler = assembler if assembler is not None else ProfileAssembler()

    def run(
        self, profile: ObjectProfile, data_dir: Path, max_image_pixels: int = 40_000_000,
        *, inspected_images: Mapping[UUID, InspectedImage] | None = None,
        artifact_store: ObjectTransaction | None = None,
        segmentation_boxes: Mapping[UUID, NormalizedBox] | None = None,
    ) -> ObjectProfile:
        if max_image_pixels <= 0:
            raise ValueError("图片像素限制必须大于零")
        current = ObjectProfile.model_validate(profile.model_dump())
        if artifact_store is not None and (
            artifact_store.data_dir.resolve() != Path(data_dir).resolve()
            or artifact_store.object_id != current.object_id
        ):
            raise ValueError("产物存储不属于当前处理事务")
        for processor in self.processors:
            snapshot = current.model_copy(deep=True)
            context = ProcessingContext(
                images=tuple(snapshot.images),
                data_dir=Path(data_dir),
                profile=snapshot,
                previous_results=tuple(snapshot.processing_results),
                max_image_pixels=max_image_pixels,
                inspected_images=inspected_images or {},
                artifact_store=artifact_store,
                segmentation_boxes=segmentation_boxes or {},
            )
            try:
                result = processor.process(context)
                if not isinstance(result, ProcessingResult):
                    raise TypeError("模块必须返回 ProcessingResult")
                result = ProcessingResult.model_validate(result.model_dump())
                if result.module_name != processor.name:
                    raise ValueError("模块结果名称与注册名称不一致")
                if result.module_name == "object_segmentation":
                    from app.models.segmentation import SegmentationData

                    payload = SegmentationData.model_validate(result.data)
                    known_images = {image.asset_id: image for image in current.images}
                    if (payload.input_count != len(known_images)
                            or {image.asset_id for image in payload.images} != set(known_images)):
                        raise ValueError("分割输入数量与档案不一致")
                    for image in payload.images:
                        if image.asset_id not in known_images or image.filename != known_images[image.asset_id].filename:
                            raise ValueError("分割结果不属于当前图片资源")
                        for mask in image.stored_masks():
                            resolve_mask_path(
                                Path(data_dir), current.object_id, image.asset_id, mask.mask_path,
                                expected_size=(image.width, image.height), max_pixels=max_image_pixels,
                                expected_bbox=mask.bbox, expected_area_ratio=mask.area_ratio,
                            )
            except Exception:
                error_id = uuid4()
                logger.exception("模块执行失败 module=%s error_id=%s", processor.name, error_id)
                result = ProcessingResult(
                    module_name=processor.name,
                    status="failed",
                    errors=[ProcessingError(
                        code="module_execution_failed",
                        message="模块执行失败，已保留其他有效结果，请根据错误编号检查服务日志。",
                        error_id=error_id,
                    )],
                )
            current = self.assembler.merge_result(current, result)
        return self.assembler.finalize(current)
