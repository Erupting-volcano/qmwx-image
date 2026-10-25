import json
from pathlib import PurePosixPath
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.image_asset import ImageAsset
from app.models.object_profile import (
    ComponentInfo, ComponentRelation, GeometryInfo, MaterialCandidate, Measurement, ObjectProfile,
)
from app.models.processing_result import ProcessingResult
from app.services.profile_assembler import ProfileAssembler


def test_profile_defaults_preserve_unknown_information():
    profile = ProfileAssembler().create_profile("待数字化物品", [])
    document = json.loads(profile.model_dump_json())

    assert document["schema_version"] == "0.1.0"
    assert document["category"] is None
    assert document["materials"] == []
    assert document["components"] == []
    assert document["geometry"]["scale_status"] == "uncalibrated"
    assert document["geometry"]["model_status"] == "not_started"
    assert document["geometry"]["mesh_path"] is None
    assert document["geometry"]["point_cloud_path"] is None
    assert document["geometry"]["cad_path"] is None
    assert document["missing_information"]
    assert profile.created_at.tzinfo is not None
    assert ObjectProfile.model_validate_json(profile.model_dump_json()) == profile


def test_unknown_measurement_is_not_an_estimate():
    measurement = Measurement(name="length")
    assert measurement.value is None
    assert measurement.unit is None
    assert measurement.source == "unknown"
    assert measurement.status == "unknown"
    assert measurement.uncertainty is None
    assert measurement.verified_by_user is False
    assert GeometryInfo().scale_status == "uncalibrated"


def test_image_asset_roundtrip_and_relative_path():
    object_id, asset_id = uuid4(), uuid4()
    asset = ImageAsset(
        asset_id=asset_id,
        filename="正面.png",
        storage_path=f"objects/{object_id}/images/{asset_id}.png",
        mime_type="image/png",
        width=32,
        height=24,
        size_bytes=100,
    )
    assert not PurePosixPath(asset.storage_path).is_absolute()
    assert ImageAsset.model_validate_json(asset.model_dump_json()) == asset


@pytest.mark.parametrize("status", ["success", "partial", "skipped", "failed", "needs_input"])
def test_processing_result_supports_required_statuses(status):
    result = ProcessingResult(module_name="example", status=status, data={})
    assert result.model_dump(mode="json")["status"] == status
    assert result.warnings == []
    assert result.errors == []


def test_invalid_status_is_rejected():
    with pytest.raises(ValidationError):
        ProcessingResult(module_name="example", status="invented", data={})


def test_estimated_dimension_does_not_become_confirmed():
    measurement = Measurement(name="hole_diameter", value=8.5, unit="mm",
                              source="algorithm_estimate", status="estimated", uncertainty=1.2)
    assert measurement.status == "estimated"
    assert not measurement.verified_by_user
    assert measurement.verification_method is None


def test_confirmed_dimension_requires_explicit_verification():
    fields = dict(name="thickness", value=2.0, unit="mm", source="user_measurement", status="confirmed")
    with pytest.raises(ValidationError):
        Measurement(**fields)
    assert Measurement(**fields, verified_by_user=True).status == "confirmed"
    independent = Measurement(**fields, verification_method="两次独立卡尺测量相符")
    assert not independent.verified_by_user
    with pytest.raises(ValidationError):
        Measurement(name="unknown", verified_by_user=True)


def test_geometry_readiness_is_separate_from_engineering_validation():
    appearance = GeometryInfo(mesh_path="objects/example/model.obj", model_status="ready", mesh_role="appearance")
    assert appearance.scale_status == "uncalibrated"
    assert appearance.engineering_status == "not_validated"
    with pytest.raises(ValidationError):
        GeometryInfo(mesh_path="objects/example/model.obj", model_status="ready", engineering_status="validated")
    validated = GeometryInfo(
        mesh_path="objects/example/model.obj", model_status="ready", mesh_role="engineering",
        scale_status="calibrated", scale_reference="用户实测孔距 50 mm",
        engineering_status="validated", verification_method="独立测量关键孔距并核对误差",
    )
    assert validated.engineering_status == "validated"


@pytest.mark.parametrize("path", ["../escape.obj", "/absolute.obj", "C:/private.obj", "a\\b.obj"])
def test_geometry_paths_cannot_escape_data_directory(path):
    with pytest.raises(ValidationError):
        GeometryInfo(mesh_path=path, model_status="ready")


@pytest.mark.parametrize("model,fields", [
    (MaterialCandidate, {"name": "纸板", "source": "algorithm_estimate"}),
    (ComponentInfo, {"name": "盒盖", "source": "algorithm_estimate"}),
    (ComponentRelation, {"from_component_id": uuid4(), "to_component_id": uuid4(),
                         "relation_type": "connected_to", "source": "algorithm_estimate"}),
])
def test_confirmed_candidates_require_verification_basis(model, fields):
    with pytest.raises(ValidationError):
        model(**fields, status="confirmed")
    assert model(**fields, status="confirmed", verification_method="用户对照实物确认").status == "confirmed"


def test_component_relations_reference_components_in_the_same_profile():
    body, lid = ComponentInfo(name="盒身"), ComponentInfo(name="盒盖")
    relation = ComponentRelation(from_component_id=body.component_id, to_component_id=lid.component_id,
                                 relation_type="connected_to", status="estimated", source="algorithm_estimate")
    profile = ObjectProfile(name="有部件的物体", components=[body, lid], component_relations=[relation])
    assert ObjectProfile.model_validate_json(profile.model_dump_json()) == profile
    with pytest.raises(ValidationError):
        ObjectProfile(name="缺失部件", components=[body], component_relations=[relation])
