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

if TYPE_CHECKING:
    from app.modules.image_metadata import InspectedImage

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
    ) -> ObjectProfile:
        if max_image_pixels <= 0:
            raise ValueError("图片像素限制必须大于零")
        current = ObjectProfile.model_validate(profile.model_dump())
        for processor in self.processors:
            snapshot = current.model_copy(deep=True)
            context = ProcessingContext(
                images=tuple(snapshot.images),
                data_dir=Path(data_dir),
                profile=snapshot,
                previous_results=tuple(snapshot.processing_results),
                max_image_pixels=max_image_pixels,
                inspected_images=inspected_images or {},
            )
            try:
                result = processor.process(context)
                if not isinstance(result, ProcessingResult):
                    raise TypeError("模块必须返回 ProcessingResult")
                result = ProcessingResult.model_validate(result.model_dump())
                if result.module_name != processor.name:
                    raise ValueError("模块结果名称与注册名称不一致")
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
