"""Typed observations validate provenance, uncertainty and counts, not VLM accuracy."""

from uuid import uuid4

from pydantic import ValidationError
import pytest

from app.models.object_understanding import (
    AppearanceInfo, CategoryCandidate, ComponentObservation, FusionData, ImageObservation,
    ImageUnderstandingState, MaterialObservation, ModelCallMetadata, ObjectUnderstandingData,
    ObservationConflict, SourcedFact, validate_fusion_against_observations, validate_sources,
)


def evidence(asset, **fields):
    return {"evidence": "图中可见的依据", "asset_ids": [str(asset)], **fields}


def observation(asset, *, component="把手", material="陶瓷", category="杯子"):
    return ImageObservation(
        asset_id=asset,
        category_candidates=[CategoryCandidate(**evidence(asset, category=category, broad_category="容器"))],
        material_candidates=[MaterialObservation(**evidence(asset, material_name=material))],
        visible_components=[ComponentObservation(**evidence(asset, component_name=component, description="侧面的环形部分"))],
        task_statuses={"classification": "success", "material": "success", "components": "success"},
    )


@pytest.mark.parametrize("model,fields", [
    (SourcedFact, {"text": "白色"}), (CategoryCandidate, {"category": "杯子"}),
    (MaterialObservation, {"material_name": "陶瓷"}),
    (ComponentObservation, {"component_name": "把手", "description": "可见环形部分"}),
])
def test_model_predictions_never_confirm_facts(model, fields):
    with pytest.raises(ValidationError):
        model(**evidence(uuid4(), status="confirmed", **fields))


def test_material_confidence_remains_candidate_and_has_uncertainty():
    item = MaterialObservation(**evidence(uuid4(), material_name="不锈钢", confidence=0.99))
    assert item.status == "candidate" and item.uncertainty
    with pytest.raises(ValidationError):
        MaterialObservation(**evidence(uuid4(), material_name="金属", confidence=1.01))


def test_foreign_sources_are_rejected_at_image_boundary():
    asset, foreign = uuid4(), uuid4()
    with pytest.raises(ValidationError):
        ImageObservation(asset_id=asset, material_candidates=[MaterialObservation(**evidence(foreign, material_name="玻璃"))])


@pytest.mark.parametrize("changes", [{"asset_ids": []}, {"evidence": "   "}, {"text": "   "}])
def test_sources_and_evidence_cannot_be_empty(changes):
    with pytest.raises(ValidationError):
        SourcedFact(**{**evidence(uuid4(), text="可见颜色"), **changes})


def test_duplicate_source_ids_and_unobserved_appearance_are_rejected():
    asset = uuid4()
    with pytest.raises(ValidationError):
        SourcedFact(**{**evidence(asset, text="红色"), "asset_ids": [asset, asset]})
    with pytest.raises(ValidationError):
        AppearanceInfo(shape_description=SourcedFact(**evidence(asset, text="可能圆形", status="candidate")))


def test_dimensions_and_hidden_geometry_are_not_output_fields():
    with pytest.raises(ValidationError):
        ImageObservation(asset_id=uuid4(), measurements=[{"name": "length", "value": 20, "unit": "cm"}])
    with pytest.raises(ValidationError):
        FusionData(geometry={"model_status": "ready"})


def test_single_view_feature_and_all_sources_survive_semantic_alias_fusion():
    first, second = uuid4(), uuid4()
    observations = [observation(first, component="把手"), observation(second, component="杯柄")]
    fusion = FusionData(
        object_category=CategoryCandidate(evidence="两图均见杯形", asset_ids=[first, second], category="杯子"),
        components=[ComponentObservation(
            evidence="两个措辞描述相同可见部件", asset_ids=[first, second], component_name="把手", description="环状侧把",
            source_labels=[{"label": "杯柄", "asset_ids": [second]}],
        )],
        appearance_summary=[SourcedFact(**evidence(first, text="此图可见一个缺口"))],
        cross_view_consensus=[SourcedFact(text="均可见杯状外形", evidence="两图共同支持", asset_ids=[first, second])],
    )
    assert validate_fusion_against_observations(fusion, observations) is fusion
    assert fusion.appearance_summary[0].asset_ids == [first]
    assert set(fusion.components[0].asset_ids) == {first, second}
    assert fusion.components[0].source_labels[0].label == "杯柄"


@pytest.mark.parametrize("kind", ["new_label", "false_alias", "wrong_alias_source", "invented_material"])
def test_fusion_rejects_unbacked_labels_or_aliases(kind):
    first, second = uuid4(), uuid4()
    observations = [observation(first, component="把手"), observation(second, component="杯柄")]
    component = evidence(first, component_name="把手", description="环状部件")
    fusion_fields = {"components": [component]}
    if kind == "new_label":
        component["component_name"] = "内部螺钉"
    elif kind == "false_alias":
        component["asset_ids"] = [first, second]
        component["source_labels"] = [{"label": "隐藏电路", "asset_ids": [second]}]
    elif kind == "wrong_alias_source":
        component["asset_ids"] = [first, second]
        component["source_labels"] = [{"label": "杯柄", "asset_ids": [first]}]
    else:
        fusion_fields = {"material_candidates": [evidence(first, material_name="钛合金")]}
    with pytest.raises(ValueError):
        validate_fusion_against_observations(FusionData.model_validate(fusion_fields), observations)


def test_cross_view_conflict_keeps_both_claims_instead_of_choosing_a_winner():
    first, second = uuid4(), uuid4()
    conflict = ObservationConflict(field="颜色", description="光照可能导致差异", claims=[
        SourcedFact(**evidence(first, text="表面偏白")), SourcedFact(**evidence(second, text="表面偏黄")),
    ])
    fusion = FusionData(conflicting_observations=[conflict])
    validate_sources(fusion, {first, second})
    assert len(fusion.conflicting_observations[0].claims) == 2
    with pytest.raises(ValidationError):
        ObservationConflict(field="颜色", description="同图不能伪造多视角", claims=[
            SourcedFact(**evidence(first, text="偏白")), SourcedFact(**evidence(first, text="偏黄")),
        ])
    with pytest.raises(ValidationError):
        FusionData(cross_view_consensus=[SourcedFact(**evidence(first, text="只有一张"))])


def test_fusion_cannot_upgrade_candidate_components_to_observed():
    first, second = uuid4(), uuid4()
    views = [observation(first), observation(second, component="杯柄")]
    views[1].visible_components[0].status = "candidate"
    candidate = ComponentObservation(evidence="两种措辞合并", component_name="把手", description="环状部分",
                                     asset_ids=[first, second], status="candidate",
                                     source_labels=[{"label": "杯柄", "asset_ids": [second]}])
    fusion = FusionData(components=[candidate])
    assert validate_fusion_against_observations(fusion, views) is fusion
    candidate.status = "observed"
    with pytest.raises(ValueError, match="候选部件"):
        validate_fusion_against_observations(fusion, views)


def test_counts_states_and_successful_reuse_fingerprints_are_consistent():
    first, second = uuid4(), uuid4()
    image = observation(first)
    data = ObjectUnderstandingData(
        input_count=2, checked_count=1, failed_count=1, image_observations=[image],
        image_states=[ImageUnderstandingState(asset_id=first, status="partial"),
                      ImageUnderstandingState(asset_id=second, status="failed", error_id=uuid4(), message="测试错误")],
    )
    assert data.checked_count == 1 and data.failed_count == 1
    for change in ({"input_count": 1}, {"checked_count": 2}, {"failed_count": 0}, {"reused_count": 2}):
        with pytest.raises(ValidationError):
            ObjectUnderstandingData.model_validate({**data.model_dump(), **change})
    with pytest.raises(ValidationError):
        ImageObservation(asset_id=first, task_statuses={"material": "failed"}, task_fingerprints={"material": "a" * 64})


def test_empty_and_old_style_defaults_are_explicitly_unknown():
    data = ObjectUnderstandingData()
    assert data.object_category is None and data.fusion_status == "not_attempted"
    assert data.image_observations == [] and data.model_metadata == []
    with pytest.raises(ValidationError):
        ObjectUnderstandingData(fusion_status="success")


def test_per_task_notes_survive_without_changing_other_task_versions_or_status():
    asset = uuid4()
    image = ImageObservation(asset_id=asset, task_statuses={"appearance": "success", "material": "failed"},
                             task_unknown_information={"appearance": ["背面不可见"], "material": ["材料未知"]},
                             task_warnings={"material": ["测试输出无效"]},
                             task_fingerprints={"appearance": "a" * 64})
    restored = ImageObservation.model_validate(image.model_dump(mode="json"))
    assert restored.task_unknown_information["appearance"] == ["背面不可见"]
    assert restored.task_warnings["material"] == ["测试输出无效"]
    assert "material" not in restored.task_fingerprints


@pytest.mark.parametrize("metadata", [{"raw": b"bytes"}, {"raw": "x" * 20000}, {"time": float("inf")}])
def test_model_specific_metadata_cannot_store_binary_or_unbounded_data(metadata):
    with pytest.raises(ValidationError):
        ModelCallMetadata(model_id="fixture", metadata=metadata)
