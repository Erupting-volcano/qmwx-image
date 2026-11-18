"""COLMAP backend behaviour against a stand-in executable; no engine required."""

from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

import pytest

from app.models.reconstruction import ReconstructionRequest
from app.modules.reconstruction.base import (
    ReconstructionBackendError, ReconstructionCancelled, ReconstructionWorkspace,
)
from app.modules.reconstruction.colmap_backend import ColmapBackend

FAKE = Path(__file__).with_name("fake_colmap.py")


def backend(**overrides):
    options = {"command": (sys.executable, str(FAKE))}
    options.update(overrides)
    return ColmapBackend(**options)


def workspace_for(tmp_path, images=3, masks=0):
    root = tmp_path / "work"
    images_dir = root / "images"
    masks_dir = root / "masks"
    output_dir = root / "output"
    for directory in (images_dir, masks_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    image_files = []
    for index in range(images):
        path = images_dir / f"{index:03d}.png"
        path.write_bytes(b"image")
        image_files.append(path)
    mask_files = []
    for index in range(min(masks, images)):
        path = masks_dir / f"{index:03d}.png"
        path.write_bytes(b"mask")
        mask_files.append(path)
    return ReconstructionWorkspace(root=root, images_dir=images_dir, masks_dir=masks_dir,
                                   output_dir=output_dir, image_files=tuple(image_files),
                                   mask_files=tuple(mask_files))


def request_for(images=3):
    return ReconstructionRequest(object_id=uuid4(), asset_ids=[uuid4() for _ in range(images)],
                                 backend_id="colmap")


def run(instance, workspace, request=None, cancel=lambda: False, reports=None):
    request = request or request_for(len(workspace.image_files))
    collect = reports.append if reports is not None else (lambda *_: None)
    return instance.run(workspace, request, lambda stage, progress: collect((stage, progress)), cancel)


def test_happy_path_parses_the_model(tmp_path):
    output = run(backend(), workspace_for(tmp_path))

    assert output.sparse.registered_images == 3
    assert output.sparse.points == 12480
    assert output.sparse.mean_reprojection_error == 0.83
    assert output.point_cloud_file.is_file()
    assert output.cameras_file.is_file()
    assert output.point_count == 12480
    assert output.intrinsics.fx == 3200
    assert output.intrinsics.model == "PINHOLE"


def test_progress_is_reported_for_every_stage(tmp_path):
    reports = []
    run(backend(), workspace_for(tmp_path), reports=reports)
    stages = [stage for stage, _ in reports]
    assert stages == ["feature_extraction", "matching", "mapping", "export"]
    assert [progress for _, progress in reports] == sorted(progress for _, progress in reports)


def test_masks_are_handed_to_the_feature_extractor(tmp_path):
    workspace = workspace_for(tmp_path, masks=2)
    run(backend(), workspace)
    log = (workspace.root / "logs" / "feature_extraction.log").read_text(encoding="utf-8")
    assert "--ImageReader.mask_path" in log


def test_modern_colmap_gpu_flags_are_detected(tmp_path):
    workspace = workspace_for(tmp_path)
    run(backend(), workspace)
    log = (workspace.root / "logs" / "feature_extraction.log").read_text(encoding="utf-8")
    assert "--FeatureExtraction.use_gpu" in log
    assert "--SiftExtraction.use_gpu" not in log


def test_legacy_colmap_gpu_flags_still_work(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_LEGACY_FLAGS", "1")
    workspace = workspace_for(tmp_path)
    run(backend(), workspace)
    log = (workspace.root / "logs" / "feature_extraction.log").read_text(encoding="utf-8")
    assert "--SiftExtraction.use_gpu" in log


def test_failed_stage_reports_the_log_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_FAIL", "mapper")
    with pytest.raises(ReconstructionBackendError, match="mapping"):
        run(backend(), workspace_for(tmp_path))


def test_missing_model_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_NO_MODEL", "1")
    with pytest.raises(ReconstructionBackendError, match="没有生成可用模型"):
        run(backend(), workspace_for(tmp_path))


def test_unparsable_statistics_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_BAD_ANALYZER", "1")
    with pytest.raises(ReconstructionBackendError, match="无法解析重建统计"):
        run(backend(), workspace_for(tmp_path))


def test_unknown_camera_model_keeps_the_result_but_not_intrinsics(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_CAMERA_MODEL", "WEIRD")
    output = run(backend(), workspace_for(tmp_path))
    assert output.intrinsics is None
    assert any("相机模型" in warning for warning in output.warnings)


def test_radial_camera_model_is_reported_as_such(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_CAMERA_MODEL", "SIMPLE_RADIAL")
    output = run(backend(), workspace_for(tmp_path))
    assert output.intrinsics.model == "SIMPLE_RADIAL"
    assert output.intrinsics.fx == 3200
    assert output.intrinsics.distortion == [0.01]


def test_unregistered_images_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_IMAGES", "2")
    output = run(backend(), workspace_for(tmp_path, images=3))
    assert any("未能注册" in warning for warning in output.warnings)


def test_cancel_stops_a_running_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_SLEEP", "5")
    cancelled = threading.Event()
    outcome = {}

    def target():
        try:
            run(backend(), workspace_for(tmp_path), cancel=cancelled.is_set)
            outcome["result"] = "finished"
        except ReconstructionCancelled:
            outcome["result"] = "cancelled"
        except Exception as exc:  # pragma: no cover - surfaced by the assert below
            outcome["result"] = repr(exc)

    thread = threading.Thread(target=target)
    thread.start()
    time.sleep(0.5)
    cancelled.set()
    thread.join(timeout=20)
    assert outcome.get("result") == "cancelled"


def test_timeout_kills_the_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_SLEEP", "5")
    with pytest.raises(ReconstructionBackendError, match="超时"):
        run(backend(timeout_seconds=0.3), workspace_for(tmp_path))


def test_cancel_before_start_does_not_launch_any_command(tmp_path):
    workspace = workspace_for(tmp_path)
    with pytest.raises(ReconstructionCancelled):
        run(backend(), workspace, cancel=lambda: True)
    assert not (workspace.root / "logs").exists()


def test_validate_rejects_bad_input_without_touching_the_engine():
    instance = backend()
    few = ReconstructionRequest(object_id=uuid4(), asset_ids=[uuid4()], backend_id="colmap")
    assert any("照片太少" in problem for problem in instance.validate(few, None))

    many = ReconstructionRequest(object_id=uuid4(), asset_ids=[uuid4() for _ in range(200)],
                                 backend_id="colmap")
    assert any("照片过多" in problem for problem in instance.validate(many, None))


def test_validate_reports_a_missing_engine():
    instance = ColmapBackend(command=("definitely-not-colmap-binary",))
    request = ReconstructionRequest(object_id=uuid4(), asset_ids=[uuid4() for _ in range(10)],
                                    backend_id="colmap")
    assert any("未找到 COLMAP" in problem for problem in instance.validate(request, None))
