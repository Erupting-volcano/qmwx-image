"""Evidence-backed visual observations; AI output can never confirm a fact."""

from collections.abc import Iterable
import unicodedata
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue, StringConstraints, field_validator, model_validator

from app.models._common import ContractModel
from app.models.segmentation import validate_model_extensions


TaskId = Literal["classification", "appearance", "material", "components", "viewpoint", "multi_view_fusion"]
ImageTaskId = Literal["classification", "appearance", "material", "components", "viewpoint"]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TaskNotes = Annotated[list[ShortText], Field(max_length=100)]
IMAGE_TASK_IDS = ("classification", "appearance", "material", "components", "viewpoint")


class SourcedRecord(ContractModel):
    evidence: Description
    asset_ids: list[UUID] = Field(min_length=1, max_length=100)

    @field_validator("asset_ids")
    @classmethod
    def unique_sources(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("来源图片不得重复")
        return value


class SourcedFact(SourcedRecord):
    text: Description
    status: Literal["observed", "candidate"] = "observed"


class SourceLabel(ContractModel):
    label: ShortText
    asset_ids: list[UUID] = Field(min_length=1, max_length=100)

    @field_validator("asset_ids")
    @classmethod
    def unique_sources(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("标签来源不得重复")
        return value


class CategoryCandidate(SourcedRecord):
    category: ShortText
    broad_category: ShortText | None = None
    status: Literal["candidate"] = "candidate"
    confidence: float | None = Field(default=None, ge=0, le=1,
                                    description="Uncalibrated model score, not a confirmed probability.")
    source_labels: list[SourceLabel] = Field(default_factory=list, max_length=30)


class ColorObservation(SourcedFact):
    role: Literal["primary", "secondary", "distribution"] = "primary"


class AppearanceInfo(ContractModel):
    colors: list[ColorObservation] = Field(default_factory=list, max_length=30)
    shape_description: SourcedFact | None = None
    surface_description: SourcedFact | None = None
    visible_features: list[SourcedFact] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def observed_appearance_only(self) -> "AppearanceInfo":
        facts = [*self.colors, *self.visible_features,
                 *[item for item in (self.shape_description, self.surface_description) if item is not None]]
        if any(item.status != "observed" for item in facts):
            raise ValueError("外观只记录可见观察，不用候选推测补满字段")
        return self


class MaterialObservation(SourcedRecord):
    material_name: ShortText
    confidence: float | None = Field(default=None, ge=0, le=1,
                                    description="Uncalibrated model score; material remains candidate.")
    status: Literal["candidate"] = "candidate"
    uncertainty: ShortText = "仅凭照片提出候选，尚未经过材料验证。"
    source_labels: list[SourceLabel] = Field(default_factory=list, max_length=30)


class ComponentObservation(SourcedRecord):
    component_name: ShortText
    description: Description
    status: Literal["observed", "candidate"] = "observed"
    source_labels: list[SourceLabel] = Field(default_factory=list, max_length=30)


class ViewpointObservation(SourcedRecord):
    viewpoint: Literal["front", "back", "left", "right", "top", "bottom", "close_up", "perspective", "unknown"]
    status: Literal["candidate"] = "candidate"


class TaskOutput(ContractModel):
    unknown_information: list[ShortText] = Field(default_factory=list, max_length=100)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)


class ClassificationData(TaskOutput):
    category_candidates: list[CategoryCandidate] = Field(default_factory=list, max_length=20)
    object_name: SourcedFact | None = None
    possible_uses: list[SourcedFact] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def uses_are_candidates(self) -> "ClassificationData":
        if any(item.status != "candidate" for item in self.possible_uses):
            raise ValueError("可能用途必须作为候选保存")
        if self.object_name is not None and self.object_name.status != "candidate":
            raise ValueError("模型提取的物体名称必须作为候选保存")
        return self


class AppearanceData(TaskOutput):
    appearance: AppearanceInfo = Field(default_factory=AppearanceInfo)


class MaterialData(TaskOutput):
    material_candidates: list[MaterialObservation] = Field(default_factory=list, max_length=30)


class ComponentsData(TaskOutput):
    visible_components: list[ComponentObservation] = Field(default_factory=list, max_length=100)


class ViewpointData(TaskOutput):
    viewpoint: ViewpointObservation


class ModelCallMetadata(ContractModel):
    model_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    model_version: str | None = Field(default=None, min_length=1, max_length=128)
    task_ids: list[TaskId] = Field(default_factory=list, max_length=6)
    processing_time: float | None = Field(default=None, ge=0)
    warnings: list[ShortText] = Field(default_factory=list, max_length=30)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def small_json_metadata(cls, value: object) -> dict:
        return validate_model_extensions({"recognition": value})["recognition"]

    @field_validator("task_ids")
    @classmethod
    def unique_tasks(cls, value: list[TaskId]) -> list[TaskId]:
        if len(value) != len(set(value)):
            raise ValueError("模型调用任务不得重复")
        return value


class ImageObservation(ContractModel):
    asset_id: UUID
    viewpoint: ViewpointObservation | None = None
    category_candidates: list[CategoryCandidate] = Field(default_factory=list, max_length=20)
    object_name: SourcedFact | None = None
    possible_uses: list[SourcedFact] = Field(default_factory=list, max_length=20)
    appearance: AppearanceInfo = Field(default_factory=AppearanceInfo)
    material_candidates: list[MaterialObservation] = Field(default_factory=list, max_length=30)
    visible_components: list[ComponentObservation] = Field(default_factory=list, max_length=100)
    observations: list[SourcedFact] = Field(default_factory=list, max_length=100)
    unknown_information: list[ShortText] = Field(default_factory=list, max_length=100)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)
    model_metadata: list[ModelCallMetadata] = Field(default_factory=list, max_length=30)
    prompt_versions: dict[ImageTaskId, ShortText] = Field(default_factory=dict)
    task_statuses: dict[ImageTaskId, Literal["success", "failed"]] = Field(default_factory=dict)
    task_fingerprints: dict[ImageTaskId, Fingerprint] = Field(default_factory=dict)
    task_unknown_information: dict[ImageTaskId, TaskNotes] = Field(default_factory=dict)
    task_warnings: dict[ImageTaskId, TaskNotes] = Field(default_factory=dict)
    analysis_fingerprint: Fingerprint | None = None

    @model_validator(mode="after")
    def single_image_sources(self) -> "ImageObservation":
        for field in (self.viewpoint, self.category_candidates, self.object_name, self.possible_uses,
                      self.appearance, self.material_candidates, self.visible_components, self.observations):
            validate_sources(field, {self.asset_id})
        ClassificationData(category_candidates=self.category_candidates, object_name=self.object_name,
                           possible_uses=self.possible_uses)
        if self.task_statuses and "success" not in self.task_statuses.values():
            raise ValueError("单图观察必须至少包含一个有效任务")
        if not set(self.task_fingerprints).issubset({key for key, status in self.task_statuses.items() if status == "success"}):
            raise ValueError("仅成功任务可记录复用指纹")
        return self


class SingleImageTaskInput(ContractModel):
    asset_id: UUID
    object_name_hint: ShortText | None = None


class FusionInput(ContractModel):
    image_observations: list[ImageObservation] = Field(min_length=1, max_length=100)
    object_name_hint: ShortText | None = None

    @model_validator(mode="after")
    def unique_images(self) -> "FusionInput":
        identifiers = [item.asset_id for item in self.image_observations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("融合输入不得重复图片")
        return self


class ObservationConflict(ContractModel):
    field: ShortText
    description: Description
    claims: list[SourcedFact] = Field(min_length=2, max_length=20)

    @model_validator(mode="after")
    def different_view_sources(self) -> "ObservationConflict":
        if len({source for claim in self.claims for source in claim.asset_ids}) < 2:
            raise ValueError("多视角冲突必须涉及至少两张来源图片")
        return self


class FusionData(TaskOutput):
    object_category: CategoryCandidate | None = None
    category_candidates: list[CategoryCandidate] = Field(default_factory=list, max_length=30)
    appearance_summary: list[SourcedFact] = Field(default_factory=list, max_length=100)
    material_candidates: list[MaterialObservation] = Field(default_factory=list, max_length=50)
    components: list[ComponentObservation] = Field(default_factory=list, max_length=100)
    cross_view_consensus: list[SourcedFact] = Field(default_factory=list, max_length=100)
    conflicting_observations: list[ObservationConflict] = Field(default_factory=list, max_length=50)
    unknown_information: list[ShortText] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def consensus_has_multiple_sources(self) -> "FusionData":
        if any(len(item.asset_ids) < 2 for item in self.cross_view_consensus):
            raise ValueError("多视角共同支持必须具有至少两张来源图片")
        return self


class ImageUnderstandingState(ContractModel):
    asset_id: UUID
    status: Literal["success", "partial", "failed", "disabled", "unavailable", "limit_exceeded"]
    error_id: UUID | None = None
    message: str = Field(default="", max_length=2000)


class ObjectUnderstandingData(FusionData):
    input_count: int = Field(default=0, ge=0, strict=True)
    checked_count: int = Field(default=0, ge=0, strict=True)
    failed_count: int = Field(default=0, ge=0, strict=True)
    skipped_count: int = Field(default=0, ge=0, strict=True)
    reused_count: int = Field(default=0, ge=0, strict=True)
    image_observations: list[ImageObservation] = Field(default_factory=list, max_length=100)
    image_states: list[ImageUnderstandingState] = Field(default_factory=list, max_length=100)
    fusion_status: Literal["not_attempted", "success", "failed", "skipped"] = "not_attempted"
    fusion_prompt_version: ShortText | None = None
    fusion_fingerprint: Fingerprint | None = None
    model_metadata: list[ModelCallMetadata] = Field(default_factory=list, max_length=100)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def consistent_inputs(self) -> "ObjectUnderstandingData":
        states = {item.asset_id: item for item in self.image_states}
        observations = {item.asset_id: item for item in self.image_observations}
        if len(states) != len(self.image_states) or len(observations) != len(self.image_observations):
            raise ValueError("理解结果不得重复引用同一图片")
        checked = {key for key, item in states.items() if item.status in {"success", "partial"}}
        failed = sum(item.status == "failed" for item in states.values())
        skipped = sum(item.status in {"disabled", "unavailable", "limit_exceeded"} for item in states.values())
        if (self.input_count != len(states) or set(observations) != checked or self.checked_count != len(checked)
                or self.failed_count != failed or self.skipped_count != skipped):
            raise ValueError("理解输入、逐图状态与有效观察数量不一致")
        if self.reused_count > self.checked_count:
            raise ValueError("复用数量不得超过有效观察数量")
        for field in (self.object_category, self.category_candidates, self.appearance_summary, self.material_candidates,
                      self.components, self.cross_view_consensus, self.conflicting_observations):
            validate_sources(field, set(observations))
        if self.fusion_status == "success":
            if not observations:
                raise ValueError("没有有效单图观察时不能声明融合成功")
            validate_fusion_against_observations(self, self.image_observations)
        return self


def validate_sources(value: object, allowed_asset_ids: Iterable[UUID | str]) -> object:
    """Reject foreign source references at each task and aggregation boundary."""
    allowed = {UUID(str(value)) for value in allowed_asset_ids}

    def visit(item: object) -> None:
        if isinstance(item, (SourcedRecord, SourceLabel)) and not set(item.asset_ids).issubset(allowed):
            raise ValueError("观察引用了不属于本次输入的来源图片")
        if isinstance(item, BaseModel):
            for name in type(item).model_fields:
                visit(getattr(item, name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return value


def _label(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_fusion_against_observations(fusion: FusionData, observations: list[ImageObservation]) -> FusionData:
    """Constrain sources/labels; semantic truth of paraphrases still needs review."""
    allowed = {item.asset_id for item in observations}
    validate_sources(fusion, allowed)
    categories, materials, components, observed_components = {}, {}, {}, {}
    for image in observations:
        for item in image.category_candidates:
            for name in (item.category, item.broad_category):
                if name:
                    categories.setdefault(_label(name), set()).update(item.asset_ids)
        for item in image.material_candidates:
            materials.setdefault(_label(item.material_name), set()).update(item.asset_ids)
        for item in image.visible_components:
            components.setdefault(_label(item.component_name), set()).update(item.asset_ids)
            if item.status == "observed":
                observed_components.setdefault(_label(item.component_name), set()).update(item.asset_ids)
    for records, attribute, known in (
            ([*fusion.category_candidates, *([fusion.object_category] if fusion.object_category else [])], "category", categories),
            (fusion.material_candidates, "material_name", materials), (fusion.components, "component_name", components)):
        for item in records:
            canonical = _label(getattr(item, attribute))
            if canonical not in known:
                raise ValueError("融合必须使用已有观察中的一个标签作为合并名称")
            sources = set(item.asset_ids)
            supported = sources & known[canonical]
            if not supported:
                raise ValueError("合并名称必须具有本条结果实际引用的单图来源")
            for alias in item.source_labels:
                alias_sources = set(alias.asset_ids)
                if not alias_sources.issubset(sources) or not alias_sources.issubset(known.get(_label(alias.label), set())):
                    raise ValueError("融合别名必须保留真实的单图标签及其来源")
                supported.update(alias_sources)
            if not sources.issubset(supported):
                raise ValueError("融合不能新增未被单图观察支持的类别、材料或部件标签及来源")
            if isinstance(item, ComponentObservation) and item.status == "observed":
                directly_observed = sources & observed_components.get(canonical, set())
                for alias in item.source_labels:
                    directly_observed.update(set(alias.asset_ids) & observed_components.get(_label(alias.label), set()))
                if not sources.issubset(directly_observed):
                    raise ValueError("融合不能将来源中的候选部件升级为直接观察")
    return fusion
