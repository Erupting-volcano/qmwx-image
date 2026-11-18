"""COLMAP command-line backend: run the engine, parse its files, never touch storage."""

from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from app.models.object_profile import ObjectProfile
from app.models.reconstruction import CameraIntrinsics, ReconstructionRequest, SparseModel
from app.modules.reconstruction.base import (
    BackendOutput, ReconstructionBackendError, ReconstructionCancelled, ReconstructionWorkspace,
)
from app.storage.local_storage import storage_io_path


MIN_IMAGES = 5
POLL_SECONDS = 0.2
# Parameters after width/height for the camera models we understand.
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": ("SIMPLE_PINHOLE", "f", "cx", "cy"),
    "PINHOLE": ("PINHOLE", "fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("SIMPLE_RADIAL", "f", "cx", "cy", "k"),
    "RADIAL": ("RADIAL", "f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("OPENCV", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
}
# COLMAP 4 prefixes these lines with its logger ("I... model.cc:439] Images: 6").
ANALYZER_PATTERNS = {
    "images": re.compile(r"\bImages:\s*(\d+)"),
    "points": re.compile(r"\bPoints:\s*(\d+)"),
    "error": re.compile(r"Mean reprojection error:\s*([0-9.]+)\s*px"),
}


class ColmapBackend:
    """Sparse reconstruction through the COLMAP CLI.

    Dense reconstruction needs CUDA, so this backend stops at the sparse model
    and point cloud: results are published as partial geometry by the service.
    """

    backend_id = "colmap"
    backend_version = "colmap-cli"

    def __init__(self, command: Sequence[str] = ("colmap",), *, timeout_seconds: float = 1800.0,
                 max_images: int = 120):
        if not command:
            raise ValueError("必须提供 COLMAP 命令")
        self.command = tuple(str(part) for part in command)
        self.timeout_seconds = timeout_seconds
        self.max_images = max_images
        self._gpu_flags: tuple[str, str] | None = None

    # -- contract ---------------------------------------------------------

    def validate(self, request: ReconstructionRequest, profile: ObjectProfile) -> list[str]:
        problems = []
        count = len(request.asset_ids)
        if count < MIN_IMAGES:
            problems.append(f"照片太少：至少 {MIN_IMAGES} 张才能做多视角重建，当前 {count} 张")
        if count > self.max_images:
            problems.append(f"照片过多：单次最多 {self.max_images} 张，当前 {count} 张")
        if not self._binary_available():
            problems.append("未找到 COLMAP 可执行文件，请安装后配置路径")
        return problems

    def run(self, workspace: ReconstructionWorkspace, request: ReconstructionRequest,
            report: Callable[[str, float], None], cancel: Callable[[], bool]) -> BackendOutput:
        started = time.monotonic()
        database = workspace.root / "database.db"
        sparse_dir = workspace.root / "sparse"
        logs = workspace.root / "logs"

        report("feature_extraction", 0.10)
        extraction_gpu_flag, matching_gpu_flag = self._gpu_options()
        self._run_tool("feature_extraction", [
            "feature_extractor", "--database_path", str(database), "--image_path", str(workspace.images_dir),
            extraction_gpu_flag, "0",
        ] + (["--ImageReader.mask_path", str(workspace.masks_dir)] if workspace.mask_files else []), cancel, logs)

        report("matching", 0.30)
        self._run_tool("matching", [
            "exhaustive_matcher", "--database_path", str(database), matching_gpu_flag, "0",
        ], cancel, logs)

        report("mapping", 0.55)
        storage_io_path(sparse_dir).mkdir(parents=True, exist_ok=True)  # COLMAP 4 requires the directory
        self._run_tool("mapping", [
            "mapper", "--database_path", str(database), "--image_path", str(workspace.images_dir),
            "--output_path", str(sparse_dir),
        ], cancel, logs)

        model = self._find_model(sparse_dir)
        if model is None:
            raise ReconstructionBackendError("没有生成可用模型：通常是照片重叠不足或纹理太少")

        report("export", 0.85)
        text_dir = workspace.root / "export"
        storage_io_path(text_dir).mkdir(parents=True, exist_ok=True)  # model_converter crashes on a missing directory
        self._run_tool("export", [
            "model_converter", "--input_path", str(model), "--output_path", str(text_dir),
            "--output_type", "TXT",
        ], cancel, logs)
        point_cloud = workspace.output_dir / "points3D.ply"
        self._run_tool("export", [
            "model_converter", "--input_path", str(model), "--output_path", str(point_cloud),
            "--output_type", "PLY",
        ], cancel, logs)

        statistics = self._analyze(model, logs)
        if not point_cloud.is_file():
            raise ReconstructionBackendError("模型已生成但点云导出失败")
        intrinsics, camera_warnings = self._read_cameras(text_dir / "cameras.txt")

        warnings = list(camera_warnings)
        if statistics["images"] < len(request.asset_ids):
            missing = len(request.asset_ids) - statistics["images"]
            warnings.append(f"有 {missing} 张照片未能注册到模型中，请检查重叠与清晰度")
        if intrinsics is None:
            warnings.append("未能解析相机内参，几何结果仍需人工核对")

        cameras_file = workspace.output_dir / "cameras.json"
        cameras_file.write_text(json.dumps({
            "intrinsics": intrinsics.model_dump(mode="json") if intrinsics else None,
            "statistics": statistics,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        return BackendOutput(
            sparse=SparseModel(input_images=len(request.asset_ids),
                               registered_images=statistics["images"],
                               points=statistics["points"],
                               mean_reprojection_error=statistics["error"]),
            intrinsics=intrinsics,
            cameras_file=cameras_file,
            point_cloud_file=point_cloud,
            point_count=statistics["points"],
            processing_seconds=round(time.monotonic() - started, 3),
            warnings=tuple(warnings),
        )

    # -- process handling -------------------------------------------------

    def _gpu_options(self) -> tuple[str, str]:
        """COLMAP 4 renamed the SIFT flags; detect once per backend instance."""
        if self._gpu_flags is None:
            extraction_help = self._help_text("feature_extractor")
            matching_help = self._help_text("exhaustive_matcher")
            extraction = ("--FeatureExtraction.use_gpu" if "--FeatureExtraction.use_gpu" in extraction_help
                          else "--SiftExtraction.use_gpu")
            matching = ("--FeatureMatching.use_gpu" if "--FeatureMatching.use_gpu" in matching_help
                        else "--SiftMatching.use_gpu")
            self._gpu_flags = (extraction, matching)
        return self._gpu_flags

    def _help_text(self, command: str) -> str:
        try:
            completed = subprocess.run([*self.command, command, "-h"], capture_output=True, text=True,
                                       timeout=60, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return ""
        return (completed.stdout or "") + (completed.stderr or "")

    def _run_tool(self, label: str, args: Sequence[str], cancel: Callable[[], bool], logs: Path) -> None:
        if cancel():
            raise ReconstructionCancelled("cancelled")
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{label}.log"
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join([*self.command, *args]) + "\n")
            log.flush()
            try:
                process = subprocess.Popen([*self.command, *args], stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, creationflags=creation_flags)
            except OSError as exc:
                raise ReconstructionBackendError(f"无法启动 COLMAP（{label}）：{exc.strerror or exc}") from None

            deadline = time.monotonic() + self.timeout_seconds
            while process.poll() is None:
                if cancel():
                    self._terminate(process)
                    raise ReconstructionCancelled("cancelled")
                if time.monotonic() > deadline:
                    self._terminate(process)
                    raise ReconstructionBackendError(f"COLMAP 阶段超时（{label}）")
                time.sleep(POLL_SECONDS)

            if process.returncode != 0:
                raise ReconstructionBackendError(
                    f"COLMAP 阶段失败（{label}，退出码 {process.returncode}）：{self._log_tail(log_path)}")

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                process.kill()
        finally:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _log_tail(log_path: Path, limit: int = 400) -> str:
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        except OSError:
            return "无日志"
        tail = " / ".join(line.strip() for line in content[-3:] if line.strip())
        return tail[-limit:] or "无日志"

    # -- parsing ----------------------------------------------------------

    def _find_model(self, sparse_dir: Path) -> Path | None:
        if not sparse_dir.is_dir():
            return None
        candidates = []
        for entry in sorted(sparse_dir.iterdir()):
            if entry.is_dir() and (entry / "cameras.bin").is_file() and (entry / "images.bin").is_file():
                candidates.append(entry)
        return candidates[0] if candidates else None

    def _analyze(self, model: Path, logs: Path) -> dict:
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / "analyzer.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            try:
                completed = subprocess.run(
                    [*self.command, "model_analyzer", "--path", str(model)],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                raise ReconstructionBackendError("COLMAP 统计阶段超时") from None
            except OSError as exc:
                raise ReconstructionBackendError(f"无法启动 COLMAP 统计：{exc.strerror or exc}") from None
        if completed.returncode != 0:
            raise ReconstructionBackendError(f"COLMAP 统计失败：{self._log_tail(log_path)}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        statistics = {}
        for name, pattern in ANALYZER_PATTERNS.items():
            match = pattern.search(text)
            if match is None:
                raise ReconstructionBackendError(f"无法解析重建统计（缺少 {name}），结果不可信")
            statistics[name] = int(match.group(1)) if name != "error" else float(match.group(1))
        if statistics["images"] <= 0:
            raise ReconstructionBackendError("重建没有注册任何照片")
        return statistics

    @staticmethod
    def _read_cameras(path: Path) -> tuple[CameraIntrinsics | None, list[str]]:
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return None, ["未找到 cameras.txt，无法读取相机内参"]
        entries = [line for line in lines if not line.startswith("#")]
        if not entries:
            return None, ["cameras.txt 为空，无法读取相机内参"]
        warnings = []
        if len(entries) > 1:
            warnings.append(f"模型包含 {len(entries)} 个相机，档案只记录第一个；多机位重建需人工核对")
        fields = entries[0].split()
        if len(fields) < 5:
            return None, ["cameras.txt 格式无法识别"]
        model_name = fields[1]
        if model_name not in CAMERA_MODELS:
            return None, [f"未识别的相机模型 {model_name}，已保留重建结果但不声明内参"]
        definition = CAMERA_MODELS[model_name]
        parameters = fields[4:]
        try:
            values = {name: float(value) for name, value in zip(definition[1:], parameters)}
            width, height = int(fields[2]), int(fields[3])
        except (ValueError, IndexError):
            return None, [f"相机模型 {model_name} 的参数无法解析"]
        if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = values["f"]
        else:
            fx, fy = values["fx"], values["fy"]
        distortion = [value for key, value in values.items() if key not in {"f", "fx", "fy", "cx", "cy"}]
        return CameraIntrinsics(model=definition[0], width=width, height=height, fx=fx, fy=fy,
                                cx=values["cx"], cy=values["cy"], distortion=distortion), warnings

    def _binary_available(self) -> bool:
        executable = self.command[0]
        if os.path.isabs(executable) or os.sep in executable:
            return Path(executable).is_file()
        return shutil.which(executable) is not None


def build_backend(binary: str | None = None) -> ColmapBackend:
    """Construct the backend from a configured path or the PATH entry."""
    command = (binary,) if binary else ("colmap",)
    return ColmapBackend(command)
