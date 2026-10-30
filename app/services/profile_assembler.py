"""Transactional, typed module merging; file persistence belongs to storage."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQualityData
from app.models.object_profile import ObjectProfile
from app.models.processing_result import ImageMetadataData, ProcessingError, ProcessingResult
from app.models.segmentation import SegmentationData

logger = logging.getLogger(__name__)

ProfileAdapter = Callable[[ObjectProfile, ProcessingResult], ObjectProfile]


@dataclass(frozen=True)
class _AdapterRegistration:
    adapter: ProfileAdapter
    editable_images: bool = False


class ProfileAssembler:
    """Register adapters to merge specific module payloads into stable profile fields."""

    def __init__(self) -> None:
        self._adapters: dict[str, _AdapterRegistration] = {
            "image_metadata": _AdapterRegistration(self._merge_image_metadata, editable_images=True),
            "image_quality": _AdapterRegistration(self._validate_image_quality),
            "object_segmentation": _AdapterRegistration(self._validate_segmentation),
        }

    def register_adapter(self, module_name: str, adapter: ProfileAdapter) -> None:
        if not module_name or not module_name.strip():
            raise ValueError("模块名称不能为空")
        if not callable(adapter):
            raise TypeError("模块合并适配器必须可调用")
        self._adapters[module_name] = _AdapterRegistration(adapter)

    def create_profile(
        self, name: str, images: list[ImageAsset], object_id: UUID | None = None,
    ) -> ObjectProfile:
        fields = {"name": name, "images": images}
        if object_id is not None:
            fields["object_id"] = object_id
        return ObjectProfile.model_validate(fields)

    @staticmethod
    def _validate_segmentation(profile: ObjectProfile, result: ProcessingResult) -> ObjectProfile:
        """Keep segmentation in module history; never turn pixel boxes into measurements."""
        payload = SegmentationData.model_validate(result.data)
        known_images = {image.asset_id: image for image in profile.images}
        if payload.input_count != len(known_images) or {image.asset_id for image in payload.images} != set(known_images):
            raise ValueError("分割结果必须对应当前档案的全部输入图片")
        if any(image.filename != known_images[image.asset_id].filename for image in payload.images):
            raise ValueError("分割结果的文件名与图片资源不一致")
        for image in payload.images:
            for mask in image.stored_masks():
                prefix = f"objects/{profile.object_id}/masks/{image.asset_id}_"
                if not mask.mask_path.startswith(prefix):
                    raise ValueError("分割掩码不属于当前物体或图片")
        return profile

    @staticmethod
    def _validate_image_quality(profile: ObjectProfile, result: ProcessingResult) -> ObjectProfile:
        """Validate quality records against this archive without changing its facts."""
        payload = ImageQualityData.model_validate(result.data)
        known_images = {image.asset_id: image for image in profile.images}
        if payload.input_count != len(known_images):
            raise ValueError("质量评估的输入数量与档案不一致")
        checked_ids = {image.asset_id for image in payload.images}
        if not checked_ids.issubset(known_images):
            raise ValueError("质量评估引用了不属于当前档案的图片")
        if any(image.filename != known_images[image.asset_id].filename for image in payload.images):
            raise ValueError("质量评估的文件名与图片资源不一致")
        if result.status == "success" and checked_ids != set(known_images):
            raise ValueError("成功的质量评估必须覆盖当前档案的所有图片")
        return profile

    @staticmethod
    def _merge_image_metadata(profile: ObjectProfile, result: ProcessingResult) -> ObjectProfile:
        payload = ImageMetadataData.model_validate(result.data)
        by_id = {item.asset_id: item for item in payload.images}
        known_ids = {image.asset_id for image in profile.images}
        if not set(by_id).issubset(known_ids):
            raise ValueError("模块返回了不属于当前档案的图片资源")
        if result.status == "success" and set(by_id) != known_ids:
            raise ValueError("成功的图片信息结果必须覆盖当前档案的所有图片")
        updated_images = []
        for image in profile.images:
            metadata = by_id.get(image.asset_id)
            if metadata is None:
                updated_images.append(image)
                continue
            fields = image.model_dump()
            fields.update(metadata.model_dump(exclude={"asset_id"}))
            updated_images.append(ImageAsset.model_validate(fields))
        profile.images = updated_images
        return profile

    @staticmethod
    def _validate_adapter_update(
        before: ObjectProfile, after: ObjectProfile, registration: _AdapterRegistration,
    ) -> ObjectProfile:
        if not isinstance(after, ObjectProfile):
            raise TypeError("合并适配器必须返回 ObjectProfile")
        # Revalidation matters: Pydantic assignment/model_copy alone does not validate fields.
        after = ObjectProfile.model_validate(after.model_dump())
        protected = (
            "schema_version", "object_id", "name", "created_at", "processing_results",
            "missing_information", "status",
        )
        for field in protected:
            if getattr(before, field) != getattr(after, field):
                raise ValueError("合并适配器不得覆盖档案身份或处理历史")
        if not registration.editable_images and before.images != after.images:
            raise ValueError("合并适配器不得覆盖原始图片资源")
        if registration.editable_images:
            before_identity = [
                (image.asset_id, image.filename, image.storage_path, image.checksum)
                for image in before.images
            ]
            after_identity = [
                (image.asset_id, image.filename, image.storage_path, image.checksum)
                for image in after.images
            ]
            if before_identity != after_identity:
                raise ValueError("图片信息适配器不得覆盖图片身份或路径")
        if after.warnings[:len(before.warnings)] != before.warnings:
            raise ValueError("合并适配器不得删除已有警告")
        return after

    def merge_result(self, profile: ObjectProfile, result: ProcessingResult) -> ObjectProfile:
        """Merge atomically, turning an invalid adapter payload into a tracked failure."""
        before = ObjectProfile.model_validate(profile.model_dump())
        accepted_result = ProcessingResult.model_validate(result.model_dump())
        candidate = before.model_copy(deep=True)
        registration = self._adapters.get(accepted_result.module_name)
        if registration is not None and accepted_result.status in {"success", "partial"}:
            try:
                candidate = self._validate_adapter_update(
                    before,
                    registration.adapter(
                        candidate, accepted_result.model_copy(deep=True),
                    ),
                    registration,
                )
            except Exception:
                error_id = uuid4()
                logger.exception(
                    "模块结果合并失败 module=%s error_id=%s",
                    accepted_result.module_name, error_id,
                )
                candidate = before.model_copy(deep=True)
                accepted_result = ProcessingResult(
                    module_name=accepted_result.module_name,
                    status="failed",
                    # Rejected mask files are discarded by the transaction; do not
                    # persist their old successful records as usable artifact links.
                    data={} if accepted_result.module_name == "object_segmentation" else accepted_result.data,
                    warnings=accepted_result.warnings,
                    errors=[*accepted_result.errors, ProcessingError(
                        code="invalid_module_data",
                        message="模块结果未通过数据校验，已保留其他有效结果。",
                        error_id=error_id,
                    )],
                )
        candidate.processing_results.append(accepted_result)
        candidate.warnings = list(dict.fromkeys([
            *candidate.warnings,
            *accepted_result.warnings,
            *[
                f"{accepted_result.module_name}：{error.message}"
                + (f"（错误编号：{error.error_id}）" if error.error_id else "")
                for error in accepted_result.errors
            ],
        ]))
        self._refresh_missing_information(candidate)
        return ObjectProfile.model_validate(candidate.model_dump())

    @staticmethod
    def _refresh_missing_information(profile: ObjectProfile) -> None:
        missing = []
        if profile.category is None:
            missing.append("category")
        if not profile.materials:
            missing.append("materials")
        if not profile.measurements or any(item.value is None for item in profile.measurements):
            missing.append("measurements")
        if profile.geometry.model_status != "ready":
            missing.append("geometry")
        if not profile.components:
            missing.append("components")
        profile.missing_information = missing

    def finalize(self, profile: ObjectProfile) -> ObjectProfile:
        completed = ObjectProfile.model_validate(profile.model_dump())
        self._refresh_missing_information(completed)
        no_active_modules = not completed.processing_results or all(
            result.status == "skipped" for result in completed.processing_results
        )
        if no_active_modules:
            completed.warnings = list(dict.fromkeys([
                *completed.warnings, "没有可执行的处理模块，已保存基础档案。",
            ]))
        has_warnings = no_active_modules or bool(completed.warnings) or any(
            result.status != "success" or result.errors or result.warnings
            for result in completed.processing_results
        )
        completed.status = "completed_with_warnings" if has_warnings else "completed"
        return ObjectProfile.model_validate(completed.model_dump())
