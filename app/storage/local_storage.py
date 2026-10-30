"""Stage a complete object and atomically publish its directory on the same disk."""

from contextlib import AbstractContextManager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
from typing import BinaryIO
from uuid import UUID, uuid4

from PIL import Image

from app.models._common import safe_relative_path
from app.models.object_profile import ObjectProfile


logger = logging.getLogger(__name__)


class ProfileNotFoundError(Exception):
    pass


class CorruptProfileError(Exception):
    pass


def _logical_path(path: Path) -> Path:
    value = os.fspath(path)
    if os.name == "nt" and value.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + value[8:])
    if os.name == "nt" and value.startswith("\\\\?\\"):
        return Path(value[4:])
    return Path(path)


def storage_io_path(path: Path, *, force: bool = False) -> Path:
    """Use Windows extended paths only for IO; never persist this representation."""
    if os.name != "nt":
        return Path(path)
    value = os.path.abspath(_logical_path(Path(path)))
    if not force and len(value.encode("utf-16-le")) // 2 < 248:
        return Path(path)
    prefix = "\\\\?\\"
    return Path(prefix + "UNC\\" + value[2:] if value.startswith("\\\\") else prefix + value)


def _storage_resolved_path(path: Path) -> Path:
    return _logical_path(storage_io_path(Path(path)).resolve())


def _long_path_failure(error: OSError, *paths: Path) -> bool:
    return (
        os.name == "nt" and getattr(error, "winerror", None) in {2, 3, 206}
        and any(storage_io_path(path) != Path(path) for path in paths)
    )


def _atomic_replace(source: Path, destination: Path) -> None:
    # Keep ordinary paths on the first call, including fault-injected replaces.
    try:
        os.replace(source, destination)
    except OSError as error:
        if not _long_path_failure(error, source, destination):
            raise
        os.replace(storage_io_path(source, force=True), storage_io_path(destination, force=True))


def _path_without_links(root: Path, relative_path: str) -> Path:
    """Resolve only descendants with no symlink or junction components."""
    root = _storage_resolved_path(Path(root))
    candidate = root
    for part in relative_path.split("/"):
        candidate = candidate / part
        physical = storage_io_path(candidate)
        if physical.is_symlink() or getattr(physical, "is_junction", lambda: False)():
            raise ValueError("产物路径不得经过符号链接")
    resolved = _storage_resolved_path(candidate)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("产物路径超出数据目录")
    return resolved


def _oriented_asset_size(data_dir: Path, object_id: UUID, asset_id: UUID) -> tuple[int, int]:
    prefix = f"objects/{object_id}/images/{asset_id}"
    originals = [Path(data_dir) / f"{prefix}{suffix}" for suffix in (".jpg", ".png", ".webp")]
    originals = [path for path in originals if storage_io_path(path).exists()]
    if len(originals) != 1:
        raise ValueError("掩码图片资源不属于当前数据目录")
    original = resolve_original_path(data_dir, object_id, asset_id, originals[0].relative_to(data_dir).as_posix())
    with Image.open(original) as image:
        size = image.size
        return size[::-1] if image.getexif().get(274) in {5, 6, 7, 8} else size


def resolve_original_path(data_dir: Path, object_id: UUID, asset_id: UUID, relative_path: str) -> Path:
    """Resolve an archive-owned original without crossing links or resource IDs."""
    parts = safe_relative_path(relative_path).split("/")
    if len(parts) != 4 or parts[:3] != ["objects", str(UUID(str(object_id))), "images"]:
        raise ValueError("原图不属于当前物体档案")
    filename = Path(parts[-1])
    if filename.stem != str(UUID(str(asset_id))) or filename.suffix not in {".jpg", ".png", ".webp"}:
        raise ValueError("原图不属于当前图片资源")
    path = storage_io_path(_path_without_links(Path(data_dir), relative_path))
    if not path.is_file():
        raise FileNotFoundError("原图文件不存在")
    return path


def resolve_mask_path(
    data_dir: Path, object_id: UUID, asset_id: UUID, relative_path: str,
    expected_size: tuple[int, int] | None = None, max_pixels: int = 40_000_000,
    expected_bbox: tuple[int, int, int, int] | None = None,
    expected_area_ratio: float | None = None,
) -> Path:
    """Validate an owned binary PNG artifact before merging or previewing it."""
    safe_path = safe_relative_path(relative_path)
    parts = safe_path.split("/")
    object_id, asset_id = UUID(str(object_id)), UUID(str(asset_id))
    if len(parts) != 4 or parts[:3] != ["objects", str(object_id), "masks"]:
        raise ValueError("掩码不属于当前物体档案")
    filename = Path(parts[-1])
    identifiers = filename.stem.split("_")
    if filename.suffix != ".png" or len(identifiers) != 2:
        raise ValueError("掩码文件名无效")
    if identifiers[0] != str(asset_id) or str(UUID(identifiers[1])) != identifiers[1]:
        raise ValueError("掩码不属于当前图片资源")
    if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels <= 0:
        raise ValueError("掩码像素上限必须为正整数")
    path = storage_io_path(_path_without_links(Path(data_dir), safe_path))
    if not path.is_file():
        raise FileNotFoundError("掩码文件不存在")
    try:
        opened = Image.open(path)
    except Image.DecompressionBombError as error:
        raise ValueError("掩码像素数超出安全限制") from error
    with opened as image:
        if image.format != "PNG" or image.mode != "L" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("掩码必须为单帧灰度 PNG")
        if image.width * image.height > max_pixels:
            raise ValueError("掩码像素数超出安全限制")
        if expected_size is not None and image.size != tuple(expected_size):
            raise ValueError("掩码尺寸与分割结果不一致")
        if image.size != _oriented_asset_size(Path(data_dir), object_id, asset_id):
            raise ValueError("掩码尺寸与转正后的原图不一致")
        image.load()
        colors = image.getcolors(maxcolors=2)
        if colors is None or any(value not in {0, 255} for _, value in colors) or image.getbbox() is None:
            raise ValueError("掩码必须包含前景，且像素仅允许0或255")
        if expected_bbox is not None and image.getbbox() != tuple(expected_bbox):
            raise ValueError("掩码边界框与分割结果不一致")
        if expected_area_ratio is not None:
            area_ratio = sum(count for count, value in colors if value == 255) / (image.width * image.height)
            if not math.isclose(area_ratio, expected_area_ratio, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("掩码面积与分割结果不一致")
    return path


class ObjectTransaction(AbstractContextManager):
    def __init__(self, storage: "LocalStorage", object_id: UUID):
        self.storage = storage
        self.object_id = UUID(str(object_id))
        self.staging_base = storage.data_dir / ".staging"
        self.data_dir = self.staging_base / str(uuid4())
        self.object_dir = self.data_dir / "objects" / str(self.object_id)
        self.images_dir = self.object_dir / "images"
        self._created_masks: set[str] = set()
        self._mask_temporaries: set[str] = set()

    def __enter__(self) -> "ObjectTransaction":
        storage_io_path(self.images_dir).mkdir(parents=True, exist_ok=False)
        return self

    def stage_image(self, stream: BinaryIO, asset_id: UUID, max_bytes: int) -> tuple[Path, int, str]:
        """Copy a caller-owned stream once; OverflowError carries bytes read."""
        if max_bytes < 0:
            raise ValueError("Image byte budget cannot be negative")
        path = self.images_dir / f"{UUID(str(asset_id))}.upload"
        size_bytes = 0
        digest = hashlib.sha256()
        with storage_io_path(path).open("xb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not isinstance(chunk, bytes):
                    raise TypeError("Image stream must return bytes")
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise OverflowError(size_bytes)
                handle.write(chunk)
                digest.update(chunk)
        return storage_io_path(path), size_bytes, digest.hexdigest()

    def finalize_image(self, path: Path, image_format: str) -> str:
        """Name a verified staged image by its real format, without copying it."""
        extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
        if image_format not in extensions:
            raise ValueError("Unsupported image format")
        path = _logical_path(Path(path))
        if (storage_io_path(path).is_symlink() or _storage_resolved_path(path).parent != _storage_resolved_path(self.images_dir)
                or path.suffix != ".upload"):
            raise ValueError("Image path is outside this transaction")
        path = _storage_resolved_path(path)
        UUID(path.stem)
        destination = path.with_suffix(extensions[image_format])
        if storage_io_path(destination).exists():
            raise FileExistsError("Image already exists")
        try:
            path.rename(destination)
        except OSError as error:
            if not _long_path_failure(error, path, destination):
                raise
            storage_io_path(path, force=True).rename(storage_io_path(destination, force=True))
        return destination.relative_to(self.data_dir).as_posix()

    def save_mask(self, asset_id: UUID, mask: Image.Image) -> str:
        """Persist an oriented, binary mask inside this object transaction."""
        asset_id = UUID(str(asset_id))
        if not isinstance(mask, Image.Image) or mask.mode != "L":
            raise ValueError("掩码必须为灰度图")
        colors = mask.getcolors(maxcolors=2)
        if colors is None or any(value not in {0, 255} for _, value in colors) or mask.getbbox() is None:
            raise ValueError("掩码必须包含前景，且像素仅允许0或255")
        if mask.size != _oriented_asset_size(self.data_dir, self.object_id, asset_id):
            raise ValueError("掩码尺寸必须与转正后的原图一致")
        relative_directory = f"objects/{self.object_id}/masks"
        directory = _path_without_links(self.data_dir, relative_directory)
        storage_io_path(directory).mkdir(exist_ok=True)
        artifact_id = uuid4()
        destination = directory / f"{asset_id}_{artifact_id}.png"
        temporary = directory / f".{artifact_id}.tmp"
        temporary_relative = temporary.relative_to(self.data_dir).as_posix()
        relative_path = destination.relative_to(self.data_dir).as_posix()
        self._mask_temporaries.add(temporary_relative)
        try:
            with storage_io_path(temporary).open("xb") as handle:
                mask.save(handle, format="PNG")
                handle.flush()
                os.fsync(handle.fileno())
            if storage_io_path(destination).exists():
                raise FileExistsError("掩码文件已存在")
            _atomic_replace(temporary, destination)
            self._created_masks.add(relative_path)
        finally:
            storage_io_path(temporary).unlink(missing_ok=True)
            self._mask_temporaries.discard(temporary_relative)
        return relative_path

    def _retain_referenced_masks(self, profile: ObjectProfile) -> None:
        # A failed mask write may also fail its immediate cleanup. Never publish
        # an untracked partial file merely because the processor isolated it.
        for relative_path in tuple(self._mask_temporaries):
            storage_io_path(_path_without_links(self.data_dir, relative_path)).unlink(missing_ok=True)
            self._mask_temporaries.discard(relative_path)
        retained = set()
        known_images = {image.asset_id: image for image in profile.images}
        for result in profile.processing_results:
            if result.module_name != "object_segmentation" or result.status not in {"success", "partial"}:
                continue
            from app.models.segmentation import SegmentationData

            payload = SegmentationData.model_validate(result.data)
            if payload.input_count != len(known_images) or {image.asset_id for image in payload.images} != set(known_images):
                raise ValueError("分割结果的输入与档案不一致")
            for image in payload.images:
                if image.filename != known_images[image.asset_id].filename:
                    raise ValueError("分割结果的文件名与图片资源不一致")
                if image.status in {"success", "needs_confirmation"} and image.mask_path is not None:
                    resolve_mask_path(
                        self.data_dir, self.object_id, image.asset_id, image.mask_path,
                        expected_size=(image.width, image.height),
                        max_pixels=image.width * image.height,
                        expected_bbox=image.bbox, expected_area_ratio=image.area_ratio,
                    )
                    retained.add(image.mask_path)
        for relative_path in self._created_masks - retained:
            storage_io_path(_path_without_links(self.data_dir, relative_path)).unlink(missing_ok=True)
        self._created_masks.intersection_update(retained)

    def save_profile(self, profile: ObjectProfile) -> dict:
        if profile.object_id != self.object_id:
            raise ValueError("Transaction object ID mismatch")
        # Validate the final serialized form, including all nested module output.
        payload = profile.model_dump(mode="json")
        ObjectProfile.model_validate(payload)
        self._retain_referenced_masks(profile)
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary_path = self.object_dir / f"profile.{uuid4()}.tmp"
        with storage_io_path(temporary_path).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary_path, self.object_dir / "profile.json")
        return json.loads(content)

    def commit(self) -> None:
        destination = self.storage.object_dir(self.object_id)
        storage_io_path(destination.parent).mkdir(parents=True, exist_ok=True)
        if storage_io_path(destination).exists():
            raise FileExistsError("Object ID already exists")
        # Both directories are inside data_dir: no cross-volume copy or half-published profile.
        _atomic_replace(self.object_dir, destination)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Resolve and check the exact recursive deletion target before removal.
        target = _storage_resolved_path(self.data_dir)
        allowed = _storage_resolved_path(self.staging_base)
        if target != allowed and target.is_relative_to(allowed) and storage_io_path(target).exists():
            try:
                shutil.rmtree(storage_io_path(target, force=True))
            except OSError:
                # A published profile must still be returned if only staging cleanup fails.
                logger.exception("Staging cleanup failed for transaction %s", self.data_dir.name)


class LocalStorage:
    def __init__(self, data_dir: Path):
        self.data_dir = _storage_resolved_path(Path(data_dir))

    def object_dir(self, object_id: UUID) -> Path:
        safe_id = UUID(str(object_id))
        return self.data_dir / "objects" / str(safe_id)

    def transaction(self, object_id: UUID) -> ObjectTransaction:
        return ObjectTransaction(self, object_id)

    def read_profile(self, object_id: UUID) -> tuple[ObjectProfile, bytes]:
        profile_path = self.object_dir(object_id) / "profile.json"
        try:
            content = storage_io_path(profile_path).read_bytes()
        except FileNotFoundError as exc:
            raise ProfileNotFoundError from exc
        try:
            profile = ObjectProfile.model_validate_json(content)
            if profile.object_id != object_id:
                raise ValueError("Stored object ID mismatch")
        except (ValueError, TypeError) as exc:
            raise CorruptProfileError from exc
        return profile, content
