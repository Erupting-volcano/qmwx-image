"""Versioned, validated object profile; unknown facts remain explicitly unknown."""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, field_validator, model_validator

from app.models._common import ContractModel, safe_relative_path
from app.models.image_asset import ImageAsset
from app.models.processing_result import ProcessingResult


class Measurement(ContractModel):
    name: str = Field(min_length=1, max_length=100)
    value: float | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=40)
    source: Literal["unknown", "user_measurement", "algorithm_estimate", "imported"] = "unknown"
    status: Literal["unknown", "estimated", "measured", "confirmed"] = "unknown"
    uncertainty: float | None = Field(
        default=None,
        ge=0,
        description="Absolute uncertainty in the same unit as value; not a percentage.",
    )
    verified_by_user: bool = False
    verification_method: str | None = Field(
        default=None, min_length=1, max_length=500,
        description="How the value was independently checked; a numeric prediction alone is not confirmation.",
    )

    @model_validator(mode="after")
    def check_measurement_consistency(self) -> "Measurement":
        if self.value is None:
            if self.status != "unknown" or self.source != "unknown":
                raise ValueError("未知测量值必须使用 unknown 状态和来源")
            if self.uncertainty is not None or self.verified_by_user or self.verification_method:
                raise ValueError("未知测量值不能具有误差或用户确认状态")
        else:
            if not self.unit or not self.unit.strip():
                raise ValueError("已知测量值必须指定单位")
            if self.status == "unknown" or self.source == "unknown":
                raise ValueError("已知测量值必须指定状态和来源")
        if self.verified_by_user and self.status != "confirmed":
            raise ValueError("用户确认过的测量值必须使用 confirmed 状态")
        if self.status == "confirmed" and not (
            self.verified_by_user or (self.verification_method and self.verification_method.strip())
        ):
            raise ValueError("confirmed 测量必须记录用户确认或独立验证依据")
        return self


class GeometryInfo(ContractModel):
    mesh_path: str | None = None
    point_cloud_path: str | None = None
    cad_path: str | None = None
    scale_status: Literal["uncalibrated", "estimated", "calibrated"] = "uncalibrated"
    model_status: Literal["not_started", "pending", "partial", "ready", "failed"] = "not_started"
    mesh_role: Literal["unknown", "appearance", "engineering"] = "unknown"
    scale_reference: str | None = Field(default=None, min_length=1, max_length=500)
    engineering_status: Literal["not_validated", "validated"] = "not_validated"
    verification_method: str | None = Field(default=None, min_length=1, max_length=500)
    quality_notes: list[str] = Field(default_factory=list)
    unknown_regions: list[str] = Field(default_factory=list)

    @field_validator("mesh_path", "point_cloud_path", "cad_path")
    @classmethod
    def validate_geometry_path(cls, value: str | None) -> str | None:
        return safe_relative_path(value) if value is not None else None

    @model_validator(mode="after")
    def check_model_paths(self) -> "GeometryInfo":
        has_model = any((self.mesh_path, self.point_cloud_path, self.cad_path))
        if self.model_status == "ready" and not has_model:
            raise ValueError("模型就绪时必须提供至少一个模型路径")
        if self.model_status == "not_started" and has_model:
            raise ValueError("尚未开始建模时不得声明模型路径")
        if self.scale_status == "calibrated" and not (self.scale_reference and self.scale_reference.strip()):
            raise ValueError("尺度已校准时必须记录尺度依据")
        if self.engineering_status == "validated" and (
            self.scale_status != "calibrated" or self.model_status != "ready"
            or not (self.verification_method and self.verification_method.strip())
        ):
            raise ValueError("工程几何通过验证时必须具备就绪模型、真实尺度依据和验证方法")
        return self


class MaterialCandidate(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: Literal["unknown", "user_input", "algorithm_estimate", "imported"] = "unknown"
    status: Literal["candidate", "confirmed"] = "candidate"
    verification_method: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_confirmation_basis(self) -> "MaterialCandidate":
        if self.status == "confirmed" and not (self.verification_method and self.verification_method.strip()):
            raise ValueError("已确认材料必须记录验证依据")
        return self


class ComponentInfo(ContractModel):
    component_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    source: Literal["unknown", "user_input", "algorithm_estimate", "imported"] = "unknown"
    status: Literal["unknown", "detected", "confirmed"] = "unknown"
    verification_method: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_confirmation_basis(self) -> "ComponentInfo":
        if self.status == "confirmed" and not (self.verification_method and self.verification_method.strip()):
            raise ValueError("已确认部件必须记录验证依据")
        return self


class ComponentRelation(ContractModel):
    """A minimal relation contract; no spatial structure is inferred in V0.1."""

    from_component_id: UUID
    to_component_id: UUID
    relation_type: str = Field(min_length=1, max_length=100)
    description: str | None = None
    source: Literal["unknown", "user_input", "algorithm_estimate", "imported"] = "unknown"
    status: Literal["unknown", "estimated", "confirmed"] = "unknown"
    verification_method: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_confirmation_basis(self) -> "ComponentRelation":
        if self.status == "confirmed" and not (self.verification_method and self.verification_method.strip()):
            raise ValueError("已确认关系必须记录验证依据")
        return self


def unknown_measurements() -> list[Measurement]:
    return [Measurement(name=name) for name in ("length", "width", "height")]


class ObjectProfile(ContractModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    object_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["completed", "completed_with_warnings"] = "completed_with_warnings"
    images: list[ImageAsset] = Field(default_factory=list)
    category: str | None = Field(default=None, min_length=1, max_length=200)
    materials: list[MaterialCandidate] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=unknown_measurements)
    geometry: GeometryInfo = Field(default_factory=GeometryInfo)
    components: list[ComponentInfo] = Field(default_factory=list)
    component_relations: list[ComponentRelation] = Field(default_factory=list)
    missing_information: list[str] = Field(
        default_factory=lambda: ["category", "materials", "measurements", "geometry", "components"]
    )
    processing_results: list[ProcessingResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("物体名称不能为空或包含控制字符")
        return value

    @model_validator(mode="after")
    def unique_identifiers(self) -> "ObjectProfile":
        image_ids = [image.asset_id for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("图片资源标识不得重复")
        measurement_names = [item.name for item in self.measurements]
        if len(measurement_names) != len(set(measurement_names)):
            raise ValueError("测量参数名称不得重复")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("部件标识不得重复")
        known_components = set(component_ids)
        for relation in self.component_relations:
            if relation.from_component_id not in known_components or relation.to_component_id not in known_components:
                raise ValueError("关系引用的部件必须属于当前档案")
            if relation.from_component_id == relation.to_component_id:
                raise ValueError("部件关系不得指向自身")
        return self
