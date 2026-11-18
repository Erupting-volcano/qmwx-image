"""Validated multi-view reconstruction contracts; geometry never claims more than it measured."""

import json
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from app.models._common import ContractModel, safe_relative_path


ARTIFACT_DIRECTORY = "geometry"

# Extension whitelist per artifact kind. The first entry is the canonical one.
ARTIFACT_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "point_cloud": (".ply", ".pcd"),
    "mesh": (".ply", ".obj", ".glb"),
    "cameras": (".bin", ".json", ".txt"),
}

BackendIdentifier = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ErrorCode = Annotated[str, Field(min_length=1, max_length=100)]
WarningText = Annotated[str, Field(min_length=1, max_length=1000)]

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled", "interrupted"]
JobStage = Literal["preflight", "feature_extraction", "matching", "mapping", "dense", "export"]
ResultStatus = Literal["succeeded", "partial", "failed", "canceled", "interrupted"]
ScaleStatus = Literal["uncalibrated", "estimated", "calibrated"]
ScaleMethod = Literal["none", "marker", "user_measurement", "depth"]

FAILED_RESULT_STATUSES = frozenset({"failed", "canceled", "interrupted"})


def artifact_path_for(object_id: UUID, kind: str, suffix: str) -> str:
    """Build the archive-owned relative path used by both storage and results."""
    if kind not in ARTIFACT_EXTENSIONS:
        raise ValueError("未知的产物类型")
    if suffix.lower() not in ARTIFACT_EXTENSIONS[kind]:
        raise ValueError("产物文件类型不受支持")
    return f"objects/{object_id}/{ARTIFACT_DIRECTORY}/{kind}-{uuid4()}{suffix.lower()}"


def validate_artifact_path(value: str, object_id: UUID, kind: str) -> str:
    """Accept only objects/{id}/geometry/{kind}-{uuid}.{ext} for the given kind."""
    value = safe_relative_path(value)
    prefix = f"objects/{object_id}/{ARTIFACT_DIRECTORY}/"
    if not value.startswith(prefix):
        raise ValueError("产物必须位于当前物体的几何目录")
    filename = value[len(prefix):]
    if not filename or "/" in filename:
        raise ValueError("产物路径不得包含子目录")
    stem, dot, suffix = filename.rpartition(".")
    if not dot:
        raise ValueError("产物必须具有扩展名")
    name_kind, separator, identifier = stem.partition("-")
    if not separator or name_kind != kind:
        raise ValueError("产物文件名与字段类型不一致")
    try:
        if str(UUID(identifier)) != identifier:
            raise ValueError
    except ValueError:
        raise ValueError("产物文件名无效") from None
    if f".{suffix.lower()}" not in ARTIFACT_EXTENSIONS[kind]:
        raise ValueError("产物文件类型不受支持")
    return value


def _bounded_options(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise ValueError("重建参数必须是可序列化的 JSON 对象") from None
    if len(encoded.encode("utf-8")) > 8192:
        raise ValueError("重建参数超过 8 KiB")
    return value


class CameraIntrinsics(ContractModel):
    """Pinhole-family intrinsics in oriented-image pixels."""

    model: Literal["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "UNKNOWN"] = "UNKNOWN"
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: list[float] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def principal_point_inside_image(self) -> "CameraIntrinsics":
        if not 0 <= self.cx <= self.width or not 0 <= self.cy <= self.height:
            raise ValueError("主点必须位于图像范围内")
        return self


class SparseModel(ContractModel):
    """Structure-from-motion summary; registration ratios are evidence, not decoration."""

    input_images: int = Field(gt=0, strict=True)
    registered_images: int = Field(gt=0, strict=True)
    points: int = Field(ge=0, strict=True)
    mean_reprojection_error: float = Field(ge=0)

    @model_validator(mode="after")
    def registered_images_within_input(self) -> "SparseModel":
        if self.registered_images > self.input_images:
            raise ValueError("注册成功的图片数不能超过输入图片数")
        return self


class ScaleCalibration(ContractModel):
    """Real-world scale is only calibrated when a reference is recorded."""

    status: ScaleStatus = "uncalibrated"
    method: ScaleMethod = "none"
    reference_length: float | None = Field(default=None, gt=0)
    reference_unit: Literal["mm", "cm", "m"] | None = None
    scale_factor: float | None = Field(default=None, gt=0)
    scale_reference: str | None = Field(default=None, min_length=1, max_length=500)
    residual: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def calibration_matches_evidence(self) -> "ScaleCalibration":
        if self.status == "calibrated":
            missing = [
                name for name, value in (
                    ("method", self.method != "none"),
                    ("reference_length", self.reference_length is not None),
                    ("reference_unit", self.reference_unit is not None),
                    ("scale_factor", self.scale_factor is not None),
                    ("scale_reference", self.scale_reference is not None),
                ) if not value
            ]
            if missing:
                raise ValueError(f"已校准尺度必须记录：{', '.join(missing)}")
        if self.status == "uncalibrated":
            if self.scale_factor is not None:
                raise ValueError("未校准的模型不得声明比例因子")
            if self.method != "none" and self.scale_reference is None:
                raise ValueError("未校准的模型不得声明校准方法")
        return self


class ReconstructionRequest(ContractModel):
    """A reconstruction request over already collected assets."""

    object_id: UUID
    asset_ids: list[UUID] = Field(min_length=1, max_length=200)
    backend_id: BackendIdentifier
    use_masks: bool = True
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("asset_ids")
    @classmethod
    def unique_assets(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("重建输入图片不得重复")
        return value

    @field_validator("options")
    @classmethod
    def bounded_options(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _bounded_options(value)


class ReconstructionJob(ContractModel):
    """Task-level state; a queued job can never carry execution evidence."""

    job_id: UUID = Field(default_factory=uuid4)
    object_id: UUID
    backend_id: BackendIdentifier
    status: JobStatus = "queued"
    stage: JobStage | None = None
    progress: float = Field(default=0.0, ge=0, le=1)
    input_images: int = Field(gt=0, strict=True)
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = Field(default=None, min_length=1, max_length=1000)
    warnings: list[WarningText] = Field(default_factory=list)

    @model_validator(mode="after")
    def state_is_consistent(self) -> "ReconstructionJob":
        if self.status == "queued":
            if self.started_at is not None or self.finished_at is not None:
                raise ValueError("排队中的任务不得具有开始或结束时间")
            if self.progress != 0 or self.stage is not None:
                raise ValueError("排队中的任务不得具有进度或阶段")
        if self.status == "running":
            if self.started_at is None or self.finished_at is not None:
                raise ValueError("运行中的任务必须有开始时间且不得有结束时间")
        if self.status in {"succeeded", "failed", "interrupted"} and self.started_at is None:
            raise ValueError("已结束的任务必须有开始时间")
        if self.status in {"succeeded", "failed", "canceled", "interrupted"} and self.finished_at is None:
            raise ValueError("已结束的任务必须有结束时间")
        if self.status == "succeeded" and self.progress != 1:
            raise ValueError("成功的任务进度必须为 1")
        if self.status in {"failed", "interrupted"} and self.error_code is None:
            raise ValueError("失败或中断的任务必须记录错误码")
        if self.status in {"queued", "running", "succeeded"} and self.error_code is not None:
            raise ValueError("未失败的任务不得声明错误码")
        if self.status == "canceled" and self.error_message is None and self.error_code is None:
            raise ValueError("取消的任务必须说明原因")
        timeline = [moment for moment in (self.created_at, self.started_at, self.finished_at) if moment is not None]
        if timeline != sorted(timeline):
            raise ValueError("任务时间顺序无效")
        # Cancellation is not a failure: a reason without an error code is enough.
        if self.status != "canceled" and self.error_code is None and self.error_message is not None:
            raise ValueError("错误说明必须与错误码同时出现")
        return self


class ReconstructionResult(ContractModel):
    """Terminal outcome written into the object archive; artifacts must be owned."""

    job_id: UUID
    object_id: UUID
    backend_id: BackendIdentifier
    backend_version: str | None = Field(default=None, min_length=1, max_length=128)
    status: ResultStatus = "failed"
    error_code: ErrorCode | None = None
    error_message: str | None = Field(default=None, min_length=1, max_length=1000)
    input_images: int = Field(gt=0, strict=True)
    sparse: SparseModel | None = None
    intrinsics: CameraIntrinsics | None = None
    cameras_path: str | None = None
    point_cloud_path: str | None = None
    mesh_path: str | None = None
    point_count: int | None = Field(default=None, ge=0, strict=True)
    mesh_vertices: int | None = Field(default=None, ge=0, strict=True)
    mesh_faces: int | None = Field(default=None, ge=0, strict=True)
    scale: ScaleCalibration = Field(default_factory=ScaleCalibration)
    processing_seconds: float | None = Field(default=None, ge=0)
    warnings: list[WarningText] = Field(default_factory=list)

    @model_validator(mode="after")
    def outcome_matches_artifacts(self) -> "ReconstructionResult":
        for field_name, kind in (
            ("cameras_path", "cameras"),
            ("point_cloud_path", "point_cloud"),
            ("mesh_path", "mesh"),
        ):
            value = getattr(self, field_name)
            if value is not None:
                validate_artifact_path(value, self.object_id, kind)

        artifact_paths = [self.cameras_path, self.point_cloud_path, self.mesh_path]
        has_artifacts = any(path is not None for path in artifact_paths)

        if self.status in FAILED_RESULT_STATUSES:
            if has_artifacts or self.sparse is not None:
                raise ValueError("失败或取消的结果不得声明稀疏模型或产物")
            if self.error_code is None:
                raise ValueError("失败的结果必须记录错误码")
            if self.mesh_vertices is not None or self.mesh_faces is not None or self.point_count is not None:
                raise ValueError("失败的结果不得声明几何统计")
        else:
            if self.error_code is not None:
                raise ValueError("成功的结果不得声明错误码")
            if self.sparse is None:
                raise ValueError("成功的结果必须包含稀疏重建摘要")
            if not has_artifacts:
                raise ValueError("成功的结果必须至少包含一个几何产物")
            if self.sparse.input_images != self.input_images:
                raise ValueError("稀疏重建的输入数量必须与结果一致")

        if self.status == "partial":
            if self.mesh_path is not None:
                raise ValueError("部分完成的结果不得声明网格")
            if not self.warnings:
                raise ValueError("部分完成的结果必须说明缺少什么")

        if self.mesh_path is None and (self.mesh_vertices is not None or self.mesh_faces is not None):
            raise ValueError("没有网格文件时不得声明网格统计")
        if self.point_cloud_path is None and self.point_count is not None and self.mesh_path is None:
            raise ValueError("没有点云或网格时不得声明点数")
        if self.error_code is None and self.error_message is not None:
            raise ValueError("错误说明必须与错误码同时出现")
        return self

    def to_geometry_update(self) -> dict:
        """Return the subset of GeometryInfo this outcome is allowed to own."""
        if self.status in FAILED_RESULT_STATUSES:
            model_status = "failed"
        elif self.mesh_path is not None:
            model_status = "ready"
        elif self.point_cloud_path is not None:
            model_status = "partial"
        else:
            model_status = "not_started"
        return {
            "point_cloud_path": self.point_cloud_path,
            "mesh_path": self.mesh_path,
            "scale_status": self.scale.status,
            "model_status": model_status,
            "scale_reference": self.scale.scale_reference,
            "quality_notes": list(self.warnings),
        }
