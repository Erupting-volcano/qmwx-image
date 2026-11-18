"""Contract tests: reconstruction outcomes may not claim more than they measured."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.object_profile import GeometryInfo
from app.models.reconstruction import (
    ARTIFACT_EXTENSIONS, CameraIntrinsics, ReconstructionJob, ReconstructionRequest,
    ReconstructionResult, ScaleCalibration, SparseModel, artifact_path_for, validate_artifact_path,
)
from app.storage import local_storage

START = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
END = START + timedelta(minutes=5)


def artifact(object_id, kind, suffix):
    return artifact_path_for(object_id, kind, suffix)


def sparse_model(input_images=10, registered_images=9, points=1000, error=0.9):
    return SparseModel(input_images=input_images, registered_images=registered_images,
                       points=points, mean_reprojection_error=error)


def succeeded_result(**overrides):
    object_id = overrides.pop("object_id", uuid4())
    values = dict(
        job_id=uuid4(), object_id=object_id, backend_id="colmap", status="succeeded",
        input_images=10, sparse=sparse_model(),
        cameras_path=artifact(object_id, "cameras", ".json"),
        point_cloud_path=artifact(object_id, "point_cloud", ".ply"),
    )
    values.update(overrides)
    return ReconstructionResult(**values)


def job(**overrides):
    values = dict(object_id=uuid4(), backend_id="colmap", input_images=10, created_at=START)
    values.update(overrides)
    return ReconstructionJob(**values)


def test_success_result_maps_to_an_accepted_geometry_update():
    result = succeeded_result()
    geometry = GeometryInfo(**result.to_geometry_update())
    assert geometry.model_status == "partial"
    assert geometry.point_cloud_path == result.point_cloud_path
    assert geometry.scale_status == "uncalibrated"


def test_mesh_result_becomes_a_ready_model():
    object_id = uuid4()
    result = succeeded_result(object_id=object_id,
                              mesh_path=artifact(object_id, "mesh", ".obj"),
                              mesh_vertices=1200, mesh_faces=800)
    geometry = GeometryInfo(**result.to_geometry_update())
    assert geometry.model_status == "ready"
    assert geometry.mesh_path == result.mesh_path


def test_success_requires_at_least_one_artifact():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="至少包含一个几何产物"):
        succeeded_result(object_id=object_id, cameras_path=None, point_cloud_path=None)


def test_success_requires_a_sparse_model():
    with pytest.raises(ValidationError, match="必须包含稀疏重建摘要"):
        succeeded_result(sparse=None)


def test_failed_result_cannot_claim_artifacts():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="不得声明稀疏模型或产物"):
        ReconstructionResult(job_id=uuid4(), object_id=object_id, backend_id="colmap", status="failed",
                             input_images=10, error_code="colmap_failed",
                             point_cloud_path=artifact(object_id, "point_cloud", ".ply"))


def test_failed_result_requires_an_error_code():
    with pytest.raises(ValidationError, match="必须记录错误码"):
        ReconstructionResult(job_id=uuid4(), object_id=uuid4(), backend_id="colmap",
                             status="canceled", input_images=10)


def test_failed_result_maps_to_a_failed_geometry_update():
    result = ReconstructionResult(job_id=uuid4(), object_id=uuid4(), backend_id="colmap",
                                  status="failed", input_images=10, error_code="colmap_failed")
    geometry = GeometryInfo(**result.to_geometry_update())
    assert geometry.model_status == "failed"
    assert geometry.point_cloud_path is None and geometry.mesh_path is None


def test_partial_result_must_explain_what_is_missing():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="必须说明缺少什么"):
        succeeded_result(object_id=object_id, status="partial")


def test_partial_result_cannot_declare_a_mesh():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="不得声明网格"):
        succeeded_result(object_id=object_id, status="partial",
                         warnings=["仅完成稀疏重建"],
                         mesh_path=artifact(object_id, "mesh", ".ply"))


def test_artifact_must_belong_to_the_result_object():
    object_id, other = uuid4(), uuid4()
    with pytest.raises(ValidationError, match="产物必须位于当前物体的几何目录"):
        succeeded_result(object_id=object_id,
                         point_cloud_path=artifact(other, "point_cloud", ".ply"))


def test_artifact_kind_must_match_its_field():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="与字段类型不一致"):
        succeeded_result(object_id=object_id,
                         mesh_path=artifact(object_id, "point_cloud", ".ply"))


def test_artifact_extension_must_be_allowed():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="产物文件类型不受支持"):
        succeeded_result(object_id=object_id,
                         point_cloud_path=f"objects/{object_id}/geometry/point_cloud-{uuid4()}.obj")


def test_mesh_statistics_require_a_mesh_file():
    with pytest.raises(ValidationError, match="没有网格文件时不得声明网格统计"):
        succeeded_result(mesh_vertices=10)


def test_point_count_requires_a_point_cloud_or_mesh():
    with pytest.raises(ValidationError, match="没有点云或网格时不得声明点数"):
        succeeded_result(point_cloud_path=None, point_count=100)


def test_sparse_input_count_must_match_the_result():
    object_id = uuid4()
    with pytest.raises(ValidationError, match="稀疏重建的输入数量必须与结果一致"):
        succeeded_result(object_id=object_id, sparse=sparse_model(input_images=8, registered_images=7))


def test_queued_job_cannot_carry_execution_evidence():
    with pytest.raises(ValidationError, match="不得具有开始或结束时间"):
        job(started_at=START)
    with pytest.raises(ValidationError, match="不得具有进度或阶段"):
        job(progress=0.5)


def test_running_job_requires_a_start_time():
    with pytest.raises(ValidationError, match="必须有开始时间"):
        job(status="running", progress=0.4, stage="mapping")


def test_successful_job_requires_finish_time_and_full_progress():
    with pytest.raises(ValidationError, match="必须有结束时间"):
        job(status="succeeded", started_at=START, progress=1.0)
    with pytest.raises(ValidationError, match="进度必须为 1"):
        job(status="succeeded", started_at=START, finished_at=END, progress=0.9)


def test_failed_job_requires_an_error_code():
    with pytest.raises(ValidationError, match="必须记录错误码"):
        job(status="failed", started_at=START, finished_at=END)


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_job_timeline_must_be_ordered(status):
    with pytest.raises(ValidationError, match="时间顺序无效"):
        job(status=status, started_at=END, finished_at=START,
            progress=1.0, error_code="x" if status == "failed" else None)


def test_calibrated_scale_requires_complete_evidence():
    with pytest.raises(ValidationError, match="已校准尺度必须记录"):
        ScaleCalibration(status="calibrated", method="user_measurement", scale_factor=2.0)


def test_calibrated_scale_can_be_recorded():
    scale = ScaleCalibration(status="calibrated", method="user_measurement",
                             reference_length=120.0, reference_unit="mm",
                             scale_factor=0.5, scale_reference="用户实测纸盒长边 120mm")
    assert scale.status == "calibrated"


def test_uncalibrated_scale_cannot_declare_a_factor():
    with pytest.raises(ValidationError, match="不得声明比例因子"):
        ScaleCalibration(status="uncalibrated", scale_factor=1.5)


def test_request_rejects_duplicate_or_empty_assets():
    asset_id = uuid4()
    with pytest.raises(ValidationError, match="不得重复"):
        ReconstructionRequest(object_id=uuid4(), asset_ids=[asset_id, asset_id], backend_id="colmap")
    with pytest.raises(ValidationError):
        ReconstructionRequest(object_id=uuid4(), asset_ids=[], backend_id="colmap")


def test_request_rejects_oversized_options():
    with pytest.raises(ValidationError, match="超过 8 KiB"):
        ReconstructionRequest(object_id=uuid4(), asset_ids=[uuid4()], backend_id="colmap",
                              options={"blob": "x" * 9000})


def test_camera_principal_point_must_be_inside_the_image():
    with pytest.raises(ValidationError, match="主点必须位于图像范围内"):
        CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=700, cy=240)


def test_validate_artifact_path_rejects_a_foreign_prefix():
    object_id = uuid4()
    with pytest.raises(ValueError, match="产物必须位于当前物体的几何目录"):
        validate_artifact_path(f"objects/{object_id}/masks/point_cloud-{uuid4()}.ply", object_id, "point_cloud")


def test_storage_and_contract_share_the_artifact_whitelist():
    assert local_storage.ARTIFACT_KINDS is ARTIFACT_EXTENSIONS
    assert local_storage.ARTIFACT_DIRECTORY == "geometry"
