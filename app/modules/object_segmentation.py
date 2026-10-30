"""Backend-independent box-guided masks using the existing image transaction."""

import logging
import math
from contextlib import ExitStack
from threading import Lock
from uuid import uuid4

from PIL import Image

from app.config import Settings
from app.models.image_asset import ImageAsset
from app.models.image_quality import ImageQualityData
from app.models.segmentation import NormalizedBox, SegmentationCandidate, SegmentationData, SegmentationImage
from app.models.processing_result import ProcessingError, ProcessingResult
from app.modules import image_metadata
from app.modules.base import BaseProcessor, ProcessingContext
from app.modules.image_input import load_verified_rgb
from app.modules.segmentation_backend import (
    BoxPrompt, MaskPrediction, SegmentationAPIError, SegmentationRequest,
    SegmentationUnavailable, check_request_capabilities,
)
from app.modules.segmentation_factory import create_backend, get_backend_registration


logger = logging.getLogger(__name__)
MIN_REVIEW_AREA_RATIO = 0.001
MAX_REVIEW_AREA_RATIO = 0.98
MIN_REVIEW_MASK_SCORE = 0.7
TARGET_REVIEW_NOTE = "请人工核对掩码是否覆盖所选目标；分割结果不保证目标身份或边界正确。"


class ObjectSegmentationProcessor(BaseProcessor):
    name = "object_segmentation"

    def __init__(self, settings: Settings | None = None, *, backend=None) -> None:
        self.settings = settings or Settings()
        self._backend = backend
        self._initialization_error = None
        self._backend_lock = Lock()
        if backend is None:
            try:
                registration = get_backend_registration(self.settings.segmentation_backend)
            except SegmentationUnavailable:
                registration = None
            self._identity = {
                "model_id": self.settings.segmentation_backend,
                "model_name": registration.model_name if registration else self.settings.segmentation_backend,
                "model_version": registration.model_version if registration else None,
            }
        else:
            self._identity = {
                "model_id": getattr(backend, "model_id", "injected_backend"),
                "model_name": getattr(backend, "model_name", "injected_backend"),
                "model_version": getattr(backend, "model_version", None),
            }

    @property
    def backend(self):
        """Initialize only the selected backend, never while starting the app."""
        with self._backend_lock:
            if self._initialization_error is not None:
                error_type, message = self._initialization_error
                raise error_type(message) from None
            if self._backend is None:
                try:
                    self._backend = create_backend(self.settings)
                except SegmentationUnavailable as exc:
                    # Do not retain traceback frames, images, or earlier requests.
                    self._initialization_error = (type(exc), str(exc))
                    raise type(exc)(str(exc)) from None
        return self._backend

    @staticmethod
    def _mask_geometry(mask: Image.Image, size: tuple[int, int]):
        if not isinstance(mask, Image.Image) or mask.mode != "L" or mask.size != size:
            raise ValueError("Mask must be L at the oriented source dimensions")
        histogram = mask.histogram()
        if sum(histogram[1:255]):
            raise ValueError("Mask pixels must be 0 or 255")
        return mask.getbbox(), histogram[255] / (size[0] * size[1])

    @staticmethod
    def _score(score, score_type):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            return None, None
        return float(score), score_type

    _load_rgb = staticmethod(load_verified_rgb)

    def _segment_asset(self, context: ProcessingContext, asset: ImageAsset) -> SegmentationImage:
        prompt = context.segmentation_boxes.get(asset.asset_id)
        if not self.settings.segmentation_enabled:
            return SegmentationImage(
                asset_id=asset.asset_id, filename=asset.filename, status="disabled",
                **self._identity,
                warnings=["物体分割未启用；现有图片上传与质量评估仍可使用。"],
            )
        if self._backend is None:
            registration = get_backend_registration(self.settings.segmentation_backend)
            if registration.status != "available":
                # Report unavailable selections even when no box was supplied.
                # The factory refuses these registrations without loading them.
                self.backend
        if prompt is None:
            return SegmentationImage(
                asset_id=asset.asset_id, filename=asset.filename, status="needs_input",
                **self._identity,
                warnings=["请在上传照片上框选一个目标；本版本不会根据物体名称自动定位。"],
            )
        prompt = NormalizedBox.model_validate(prompt.model_dump() if isinstance(prompt, NormalizedBox) else prompt)
        if context.artifact_store is None:
            raise SegmentationUnavailable("本次处理没有提供事务产物存储，无法安全保存分割掩码。")
        with self._load_rgb(context, asset) as rgb:
            width, height = rgb.size
            request = SegmentationRequest(rgb, BoxPrompt(prompt.to_pixels(width, height)))
            backend = self.backend
            check_request_capabilities(request, backend.capabilities)
            prediction = backend.predict(request)
        if not isinstance(prediction, MaskPrediction) or not isinstance(prediction.mask, Image.Image):
            raise ValueError("Backend must return a real mask image")
        with ExitStack() as stack:
            mask = stack.enter_context(prediction.mask)
            if len(prediction.candidates) > 16:
                raise ValueError("Too many candidate masks")
            for candidate in prediction.candidates:
                if not isinstance(candidate.mask, Image.Image):
                    raise ValueError("Candidate must contain an image")
                stack.enter_context(candidate.mask)
            if prediction.candidates and not backend.capabilities.multiple_masks:
                raise ValueError("Backend returned undeclared multiple-mask output")
            if (not backend.capabilities.quality_score and (prediction.score is not None
                    or any(candidate.score is not None for candidate in prediction.candidates))):
                raise ValueError("Backend returned an undeclared quality score")
            if (prediction.model_id is not None and prediction.model_id != self._identity["model_id"]
                    or prediction.model_version is not None
                    and prediction.model_version != self._identity["model_version"]):
                raise ValueError("Backend result identity mismatch")
            metadata = {
                **self._identity, "processing_time": prediction.processing_time,
                "extensions": prediction.extensions,
            }
            bbox, area_ratio = self._mask_geometry(mask, (width, height))
            if area_ratio == 0 or bbox is None:
                return SegmentationImage(
                    asset_id=asset.asset_id, filename=asset.filename, status="not_found",
                    **metadata,
                    prompt_type="box", prompt_box=prompt, width=width, height=height,
                    area_ratio=0, warnings=["模型没有在框选范围中生成前景，请调整目标框或补拍照片。", *prediction.warnings],
                    error_id=uuid4(),
                )
            warnings = [TARGET_REVIEW_NOTE, *prediction.warnings]
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
            score, score_type = self._score(prediction.score, prediction.score_type)
            if score is None:
                needs_review = True
                warnings.append("服务未提供可用的质量置信分数，请人工确认；不会伪造分数。")
            elif score_type != "predicted_iou":
                needs_review = True
                warnings.append("该模型分数类型不是预测IoU，已保留原始分数，请人工确认；不与其他模型的评分直接比较。")
            elif not 0 <= score <= 1:
                needs_review = True
                warnings.append("预测IoU超出0–1范围，已保留原始分数，请人工确认。")
            elif score < MIN_REVIEW_MASK_SCORE:
                needs_review = True
                warnings.append("模型给出的掩码质量分数偏低，请人工确认边界。")
            candidates = []
            for candidate in prediction.candidates:
                candidate_bbox, candidate_area = self._mask_geometry(candidate.mask, (width, height))
                if candidate_bbox is None:
                    warnings.append("一个额外候选掩码为空，未保存该候选；主要掩码仍然保留。")
                    needs_review = True
                    continue
                candidate_score, candidate_score_type = self._score(candidate.score, candidate.score_type)
                candidates.append(SegmentationCandidate(
                    mask_path=context.artifact_store.save_mask(asset.asset_id, candidate.mask),
                    bbox=candidate_bbox, area_ratio=candidate_area,
                    score=candidate_score, score_type=candidate_score_type, warnings=list(candidate.warnings),
                ))
            if candidates:
                needs_review = True
                warnings.append("模型返回了多个候选掩码，当前展示主要结果；请人工核对目标，其他候选以独立PNG保留。")
            mask_path = context.artifact_store.save_mask(asset.asset_id, mask)
        return SegmentationImage(
            asset_id=asset.asset_id, filename=asset.filename,
            status="needs_confirmation" if needs_review else "success",
            **metadata,
            prompt_type="box", prompt_box=prompt, width=width, height=height,
            mask_path=mask_path, bbox=bbox, area_ratio=area_ratio,
            confidence=score if score is not None and score_type in {None, "predicted_iou"} and 0 <= score <= 1 else None,
            score=score, score_type=score_type, candidates=candidates, warnings=warnings,
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
                message = (str(exc) if isinstance(exc, (SegmentationUnavailable, SegmentationAPIError,
                                                       image_metadata.InvalidImageError))
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
                    **self._identity,
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
