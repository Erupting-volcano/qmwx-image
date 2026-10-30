"""Task-level visual understanding; provider calls and prompts stay independent."""

from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import logging
import math
from uuid import uuid4

from app.config import Settings
from app.models.object_understanding import (
    IMAGE_TASK_IDS, FusionData, ImageObservation, ImageUnderstandingState, ModelCallMetadata,
    ObjectUnderstandingData, validate_fusion_against_observations,
)
from app.models.processing_result import ProcessingError, ProcessingResult
from app.models.segmentation import SegmentationData
from app.modules.base import BaseProcessor, ProcessingContext
from app.modules.image_input import load_verified_rgb
from app.modules.image_metadata import InvalidImageError
from app.modules.object_understanding.tasks import compose_output_schema, get_task
from app.modules.recognition_backend import (
    RecognitionCapabilities, RecognitionError, RecognitionImage, RecognitionRequest,
    RecognitionResult, RecognitionUnavailable, check_recognition_capabilities,
)
from app.modules.recognition_factory import RecognitionRouter
from app.prompts.object_understanding import compose_prompts
from app.storage.local_storage import resolve_mask_path


logger = logging.getLogger(__name__)
TASK_FIELDS = {
    "classification": ("category_candidates", "object_name", "possible_uses"),
    "appearance": ("appearance",), "material": ("material_candidates",),
    "components": ("visible_components",), "viewpoint": ("viewpoint",),
}


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class ObjectUnderstandingProcessor(BaseProcessor):
    name = "object_understanding"

    def __init__(self, settings: Settings | None = None, *, router=None):
        self.settings = settings or Settings()
        self.router = router if router is not None else RecognitionRouter(self.settings)

    @staticmethod
    def _previous(context):
        for result in reversed(context.previous_results):
            if result.module_name == "object_understanding" and result.status in {"success", "partial"}:
                try:
                    return ObjectUnderstandingData.model_validate(result.data)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _error(exc, errors, *, asset_id=None, task_id="analysis"):
        error_id = uuid4()
        message = str(exc) if isinstance(exc, (RecognitionError, InvalidImageError)) else "物体理解任务未完成，请根据错误编号检查配置或重新分析。"
        # Raw provider exceptions/JSON may contain pictures or secrets. Log IDs only.
        logger.warning("Object understanding failed task=%s error_id=%s asset_id=%s", task_id, error_id, asset_id)
        errors.append(ProcessingError(code=f"understanding_{task_id}_failed", message=message,
                                      error_id=error_id, asset_id=asset_id))
        return error_id, message

    @staticmethod
    def _metadata(result, task_ids):
        return ModelCallMetadata(model_id=result.model_id, model_version=result.model_version,
                                 task_ids=list(task_ids), processing_time=result.processing_time,
                                 warnings=[warning[:500] for warning in result.warnings[:30]], metadata=result.metadata)

    @staticmethod
    def _analyze(backend, request):
        check_recognition_capabilities(request, backend.capabilities)
        result = backend.analyze(request)
        if (not isinstance(result, RecognitionResult) or result.model_id != backend.model_id
                or result.model_version != backend.model_version):
            raise ValueError("Recognition backend returned an invalid result or identity")
        return result

    @staticmethod
    def _target_box(context, asset, size, warnings):
        if context.profile is None:
            return None
        for result in reversed(context.previous_results):
            if result.module_name != "object_segmentation":
                continue
            if result.status not in {"success", "partial"}:
                return None
            try:
                item = next((item for item in SegmentationData.model_validate(result.data).images
                             if item.asset_id == asset.asset_id), None)
                if item is None or item.mask_path is None:
                    return None
                resolve_mask_path(context.data_dir, context.profile.object_id, asset.asset_id, item.mask_path,
                                  expected_size=size, max_pixels=context.max_image_pixels,
                                  expected_bbox=item.bbox, expected_area_ratio=item.area_ratio)
                return item.bbox
            except (ValueError, OSError):
                warnings.append("辅助分割掩码未通过校验，本次仅使用原图上下文。")
                return None
        return None

    def _observe(self, context, asset, previous, errors):
        warnings, metadata, versions, statuses, fingerprints = [], [], {}, {}, {}
        task_unknown, task_warnings = {}, {}
        fields, groups = {"asset_id": asset.asset_id}, defaultdict(list)
        last_error, unavailable = None, False
        with load_verified_rgb(context, asset) as rgb:
            path = context.resolve_asset(asset)
            with path.open("rb") as handle:
                image_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            box = self._target_box(context, asset, rgb.size, warnings)
            original_size = rgb.size
            if max(rgb.size) > self.settings.recognition_max_image_side:
                rgb.thumbnail((self.settings.recognition_max_image_side, self.settings.recognition_max_image_side))
                if box is not None:
                    box = (math.floor(box[0] * rgb.width / original_size[0]),
                           math.floor(box[1] * rgb.height / original_size[1]),
                           math.ceil(box[2] * rgb.width / original_size[0]),
                           math.ceil(box[3] * rgb.height / original_size[1]))
                warnings.append("视觉理解使用保留完整画面的缩小副本，原图未修改；细节能力受分析分辨率限制。")
            for task_id in IMAGE_TASK_IDS:
                try:
                    prompt = compose_prompts((task_id,), model_id=self.router.model_id_for_task(task_id))
                    signature = _fingerprint({"image": image_hash, "box": box, "asset_id": str(asset.asset_id),
                                              "name_hint": context.profile.name if context.profile else None,
                                              "auxiliary_crop_policy": box is not None and self.settings.recognition_max_images > 1,
                                              "prompt": prompt.fingerprint,
                                              "model": self.router.signature_for_task(task_id)})
                    versions[task_id] = prompt.versions[task_id]
                    if (previous is not None and previous.task_statuses.get(task_id) == "success"
                            and previous.task_fingerprints.get(task_id) == signature):
                        for field in TASK_FIELDS[task_id]:
                            fields[field] = getattr(previous, field)
                        statuses[task_id], fingerprints[task_id] = "success", signature
                        task_unknown[task_id] = list(previous.task_unknown_information.get(task_id, []))
                        task_warnings[task_id] = list(previous.task_warnings.get(task_id, []))
                        continue
                    key = (self.router.model_id_for_task(task_id),
                           "combined" if self.settings.recognition_combine_tasks else task_id)
                    groups[key].append(task_id)
                    fingerprints[task_id] = signature
                except Exception as exc:
                    last_error = self._error(exc, errors, asset_id=asset.asset_id, task_id=task_id)
                    unavailable |= isinstance(exc, RecognitionUnavailable)
                    statuses[task_id] = "failed"
            if previous is not None and statuses:
                metadata.extend(item for item in previous.model_metadata
                                if any(task in statuses and statuses[task] == "success" for task in item.task_ids))
                warnings.extend(previous.warnings if not groups else [])
            reused = bool(statuses) and not groups and all(value == "success" for value in statuses.values())
            for task_ids in groups.values():
                try:
                    backend = self.router.get_backend(task_ids[0])
                    with ExitStack() as stack:
                        images = [RecognitionImage(asset.asset_id, rgb)]
                        if box is not None and backend.capabilities.multi_image_input and self.settings.recognition_max_images > 1:
                            crop = stack.enter_context(rgb.crop(box))
                            images.append(RecognitionImage(asset.asset_id, crop, "target_region"))
                        prompt = compose_prompts(tuple(task_ids), model_id=backend.model_id)
                        request = RecognitionRequest(
                            task_id="image_observation" if len(task_ids) > 1 else task_ids[0],
                            instruction=prompt.instruction, images=tuple(images), output_schema=compose_output_schema(task_ids),
                            context={"asset_ids": [str(asset.asset_id)], "task_ids": task_ids,
                                     "object_name_hint": context.profile.name if context.profile else None,
                                     "target_region_is_auxiliary": len(images) > 1},
                            required_capabilities=RecognitionCapabilities(image_input=True, structured_output=True),
                        )
                        result = self._analyze(backend, request)
                    metadata.append(self._metadata(result, task_ids))
                    warnings.extend(result.warnings)
                    for task_id in task_ids:
                        try:
                            data = get_task(task_id).validate_output(result.data[task_id], {asset.asset_id})
                            for field in TASK_FIELDS[task_id]:
                                fields[field] = getattr(data, field)
                            task_warnings[task_id] = list(getattr(data, "warnings", []))
                            task_unknown[task_id] = list(getattr(data, "unknown_information", []))
                            statuses[task_id] = "success"
                        except Exception as exc:
                            last_error = self._error(exc, errors, asset_id=asset.asset_id, task_id=task_id)
                            statuses[task_id] = "failed"
                except Exception as exc:
                    last_error = self._error(exc, errors, asset_id=asset.asset_id, task_id="analysis")
                    unavailable |= isinstance(exc, RecognitionUnavailable)
                    for task_id in task_ids:
                        statuses[task_id] = "failed"
            successful = {task for task, status in statuses.items() if status == "success"}
            if not successful:
                return None, ImageUnderstandingState(asset_id=asset.asset_id,
                    status="unavailable" if unavailable else "failed",
                    error_id=last_error[0] if last_error else None,
                    message=last_error[1] if last_error else "没有取得有效观察结果。"), False
            for task_id in set(IMAGE_TASK_IDS) - successful:
                fingerprints.pop(task_id, None)
                warnings.append(f"{task_id} 任务未完成，保留其他独立观察。")
            unknown = list(dict.fromkeys(item for values in task_unknown.values() for item in values))
            if len(unknown) > 100:
                warnings.append("未知事项汇总显示前100条，各任务的完整事项仍单独保留。")
            if len(metadata) > 30:
                metadata = metadata[-30:]
                warnings.append("本次展示最近30条调用元数据，更早记录仍保留在处理历史中。")
            observation = ImageObservation(
                **fields, task_statuses=statuses, task_fingerprints=fingerprints, prompt_versions=versions,
                task_unknown_information=task_unknown, task_warnings=task_warnings,
                unknown_information=unknown[:100],
                analysis_fingerprint=_fingerprint(fingerprints), model_metadata=metadata,
                warnings=list(dict.fromkeys(warning[:500] for warning in [
                    *warnings, *[item for values in task_warnings.values() for item in values]]))[:100],
            )
            return observation, ImageUnderstandingState(asset_id=asset.asset_id,
                status="success" if len(successful) == len(IMAGE_TASK_IDS) else "partial",
                error_id=last_error[0] if last_error else None, message=last_error[1] if last_error else ""), reused

    def _fuse(self, observations, previous, errors):
        if len(observations) < 2:
            if not observations:
                return FusionData(), "not_attempted", None, None, []
            image = observations[0]
            return FusionData(category_candidates=image.category_candidates,
                object_category=image.category_candidates[0] if len(image.category_candidates) == 1 else None,
                material_candidates=image.material_candidates, components=image.visible_components,
                unknown_information=image.unknown_information), "skipped", None, None, []
        task_id = "multi_view_fusion"
        try:
            prompt = compose_prompts((task_id,), model_id=self.router.model_id_for_task(task_id))
            # Exclude elapsed time/warnings from reuse identity; task inputs remain stable.
            inputs = [image.model_dump(mode="json", exclude={
                "model_metadata", "warnings", "prompt_versions", "task_fingerprints", "analysis_fingerprint",
                "task_unknown_information", "task_warnings",
            }) for image in observations]
            fingerprint = _fingerprint({"observations": inputs, "prompt": prompt.fingerprint,
                                        "model": self.router.signature_for_task(task_id)})
            if previous is not None and previous.fusion_status == "success" and previous.fusion_fingerprint == fingerprint:
                data = FusionData.model_validate({name: getattr(previous, name) for name in FusionData.model_fields})
                return data, "success", prompt.versions[task_id], fingerprint, previous.model_metadata
            backend = self.router.get_backend(task_id)
            request = RecognitionRequest(task_id=task_id, instruction=prompt.instruction,
                output_schema=get_task(task_id).schema(), context={"image_observations": inputs},
                required_capabilities=RecognitionCapabilities(structured_output=True))
            result = self._analyze(backend, request)
            data = get_task(task_id).validate_output(result.data, {image.asset_id for image in observations})
            validate_fusion_against_observations(data, observations)
            return data, "success", prompt.versions[task_id], fingerprint, [self._metadata(result, [task_id])]
        except Exception as exc:
            self._error(exc, errors, task_id=task_id)
            return FusionData(), "failed", None, None, []

    def process(self, context: ProcessingContext) -> ProcessingResult:
        observations, states, errors, warnings, reused_count = [], [], [], [], 0
        previous = self._previous(context)
        old = {item.asset_id: item for item in previous.image_observations} if previous else {}
        for index, asset in enumerate(context.images):
            if not self.settings.recognition_enabled:
                states.append(ImageUnderstandingState(asset_id=asset.asset_id, status="disabled"))
                continue
            if index >= self.settings.recognition_max_images:
                states.append(ImageUnderstandingState(asset_id=asset.asset_id, status="limit_exceeded",
                                                      message="本批理解图片数达到配置上限，原图仍已保存。"))
                warnings.append("部分图片超过本批理解数量上限，未发送模型请求。")
                continue
            try:
                observation, state, reused = self._observe(context, asset, old.get(asset.asset_id), errors)
                states.append(state)
                if observation is not None:
                    observations.append(observation)
                    reused_count += int(reused)
            except Exception as exc:
                error_id, message = self._error(exc, errors, asset_id=asset.asset_id)
                states.append(ImageUnderstandingState(asset_id=asset.asset_id, status="failed", error_id=error_id, message=message))
        fused, fusion_status, version, fingerprint, metadata = self._fuse(observations, previous, errors)
        fused_fields = fused.model_dump()
        all_warnings = list(dict.fromkeys(warning[:500] for warning in [*fused.warnings, *warnings]))
        fused_fields["warnings"] = (all_warnings if len(all_warnings) <= 100 else
                                    [*all_warnings[:99], "汇总警告仅显示前99条，逐图详情和调用信息仍保留在档案中。"])
        data = ObjectUnderstandingData(
            **fused_fields, input_count=len(states), checked_count=len(observations),
            failed_count=sum(item.status == "failed" for item in states),
            skipped_count=sum(item.status in {"disabled", "unavailable", "limit_exceeded"} for item in states),
            reused_count=reused_count, image_observations=observations, image_states=states,
            fusion_status=fusion_status, fusion_prompt_version=version, fusion_fingerprint=fingerprint,
            model_metadata=metadata,
        )
        status = ("success" if observations and len(observations) == len(states) and not errors
                  and all(item.status == "success" for item in states) and fusion_status != "failed"
                  else "partial" if observations else "failed" if any(item.status == "failed" for item in states)
                  else "skipped")
        return ProcessingResult(module_name=self.name, status=status, data=data.model_dump(mode="json"),
            warnings=[*data.warnings, *[f"图片 {image.asset_id}：{warning}" for image in observations for warning in image.warnings]],
            errors=errors)
