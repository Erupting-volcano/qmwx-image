"""Rollback-safe updates of an existing object, and geometry artifact ownership."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.object_profile import ObjectProfile
from app.storage import local_storage
from app.storage.local_storage import (
    LocalStorage, ObjectUpdateTransaction, ProfileNotFoundError, resolve_artifact_path,
)


def create_object(storage: LocalStorage, name: str = "原物体"):
    object_id = uuid4()
    with storage.transaction(object_id) as transaction:
        transaction.save_profile(ObjectProfile(name=name, object_id=object_id))
        transaction.commit()
    return object_id


def staging_entries(storage: LocalStorage):
    staging = storage.data_dir / ".staging"
    return sorted(path.name for path in staging.iterdir()) if storage_io_exists(staging) else []


def storage_io_exists(path: Path) -> bool:
    return local_storage.storage_io_path(path).exists()


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(tmp_path / "data")


def test_update_persists_new_fields_and_keeps_existing_files(storage):
    object_id = create_object(storage)
    masks = storage.object_dir(object_id) / "masks"
    masks.mkdir()
    (masks / "keep.txt").write_text("保留内容", encoding="utf-8")

    with storage.update_transaction(object_id) as transaction:
        profile = transaction.read_profile()
        profile.category = "纸盒"
        transaction.save_profile(profile)
        transaction.commit()

    saved, _ = storage.read_profile(object_id)
    assert saved.category == "纸盒"
    assert (storage.object_dir(object_id) / "masks" / "keep.txt").read_text(encoding="utf-8") == "保留内容"
    assert staging_entries(storage) == []


def test_update_clones_unchanged_files_as_hard_links(storage):
    object_id = create_object(storage)
    (storage.object_dir(object_id) / "masks").mkdir()
    payload = storage.object_dir(object_id) / "masks" / "keep.txt"
    payload.write_text("x", encoding="utf-8")

    with storage.update_transaction(object_id) as transaction:
        staged = transaction.object_dir / "masks" / "keep.txt"
        assert os.stat(local_storage.storage_io_path(staged)).st_ino == os.stat(
            local_storage.storage_io_path(payload)).st_ino
        assert staged.read_text(encoding="utf-8") == "x"


def test_failed_publish_restores_the_previous_profile(storage, monkeypatch):
    object_id = create_object(storage)
    original = local_storage._atomic_replace
    calls = {"count": 0}

    def failing_replace(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:  # profile.json written, object publication attempted
            raise OSError("simulated publish failure")
        return original(source, destination)

    monkeypatch.setattr(local_storage, "_atomic_replace", failing_replace)
    with pytest.raises(OSError, match="simulated publish failure"):
        with storage.update_transaction(object_id) as transaction:
            profile = transaction.read_profile()
            profile.category = "不应保存"
            transaction.save_profile(profile)
            transaction.commit()

    saved, _ = storage.read_profile(object_id)
    assert saved.category is None
    assert staging_entries(storage) == []


def test_update_of_missing_object_is_rejected(storage):
    with pytest.raises(ProfileNotFoundError):
        with storage.update_transaction(uuid4()):
            pass


def test_commit_requires_an_active_transaction(storage):
    object_id = create_object(storage)
    transaction = storage.update_transaction(object_id)
    with pytest.raises(RuntimeError):
        transaction.commit()


def test_writes_after_commit_are_rejected(storage, tmp_path):
    object_id = create_object(storage)
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"data")

    with storage.update_transaction(object_id) as transaction:
        transaction.save_profile(transaction.read_profile())
        transaction.commit()
        with pytest.raises(RuntimeError, match="已经提交"):
            transaction.read_profile()
        with pytest.raises(RuntimeError, match="已经提交"):
            transaction.save_profile(transaction.read_profile())
        with pytest.raises(RuntimeError, match="已经提交"):
            transaction.save_artifact("point_cloud", source, max_bytes=64)


def test_transaction_must_be_entered_before_use(storage):
    object_id = create_object(storage)
    transaction = storage.update_transaction(object_id)
    with pytest.raises(RuntimeError, match="尚未开始"):
        transaction.read_profile()
    with pytest.raises(RuntimeError, match="尚未开始"):
        transaction.save_profile(ObjectProfile(name="新名称", object_id=object_id))
    with pytest.raises(RuntimeError, match="尚未开始"):
        transaction.commit()


def test_second_commit_is_rejected(storage):
    object_id = create_object(storage)
    with storage.update_transaction(object_id) as transaction:
        transaction.save_profile(transaction.read_profile())
        transaction.commit()
        with pytest.raises(RuntimeError):
            transaction.commit()


def test_update_rejects_a_profile_from_another_object(storage):
    object_id = create_object(storage)
    with pytest.raises(ValueError, match="object ID"):
        with storage.update_transaction(object_id) as transaction:
            transaction.save_profile(ObjectProfile(name="另一个物体", object_id=uuid4()))


def test_staging_is_cleaned_when_the_body_fails(storage):
    object_id = create_object(storage)
    with pytest.raises(RuntimeError, match="boom"):
        with storage.update_transaction(object_id):
            raise RuntimeError("boom")

    saved, _ = storage.read_profile(object_id)
    assert saved.name == "原物体"
    assert staging_entries(storage) == []


def test_save_artifact_round_trip(storage, tmp_path):
    object_id = create_object(storage)
    source = tmp_path / "cloud.ply"
    source.write_bytes(b"ply\nformat ascii 1.0\n")

    with storage.update_transaction(object_id) as transaction:
        relative = transaction.save_artifact("point_cloud", source, max_bytes=1024)
        transaction.save_profile(transaction.read_profile())
        transaction.commit()

    assert relative.startswith(f"objects/{object_id}/geometry/point_cloud-")
    assert relative.endswith(".ply")
    resolved = resolve_artifact_path(storage.data_dir, object_id, relative, max_bytes=1024)
    assert resolved.read_bytes() == b"ply\nformat ascii 1.0\n"


@pytest.mark.parametrize("kind,filename", [
    ("point_cloud", "model.ply"),
    ("mesh", "model.obj"),
    ("cameras", "cameras.bin"),
])
def test_each_artifact_kind_accepts_its_extension(storage, tmp_path, kind, filename):
    object_id = create_object(storage)
    source = tmp_path / filename
    source.write_bytes(b"data")

    with storage.update_transaction(object_id) as transaction:
        relative = transaction.save_artifact(kind, source, max_bytes=64)
        transaction.save_profile(transaction.read_profile())
        transaction.commit()

    assert resolve_artifact_path(storage.data_dir, object_id, relative, max_bytes=64).exists()


def test_save_artifact_rejects_unknown_kind(storage, tmp_path):
    object_id = create_object(storage)
    source = tmp_path / "model.ply"
    source.write_bytes(b"data")

    with pytest.raises(ValueError, match="未知的产物类型"):
        with storage.update_transaction(object_id) as transaction:
            transaction.save_artifact("cad", source, max_bytes=64)


def test_save_artifact_rejects_a_wrong_extension(storage, tmp_path):
    object_id = create_object(storage)
    source = tmp_path / "model.exe"
    source.write_bytes(b"data")

    with pytest.raises(ValueError, match="产物文件类型不受支持"):
        with storage.update_transaction(object_id) as transaction:
            transaction.save_artifact("mesh", source, max_bytes=64)


def test_save_artifact_rejects_oversized_and_empty_files(storage, tmp_path):
    object_id = create_object(storage)
    big = tmp_path / "big.ply"
    big.write_bytes(b"x" * 65)
    empty = tmp_path / "empty.ply"
    empty.touch()

    with storage.update_transaction(object_id) as transaction:
        with pytest.raises(ValueError, match="超过大小上限"):
            transaction.save_artifact("point_cloud", big, max_bytes=64)
        with pytest.raises(ValueError, match="产物文件为空"):
            transaction.save_artifact("point_cloud", empty, max_bytes=64)
        geometry = transaction.object_dir / "geometry"
        assert not geometry.exists() or list(geometry.iterdir()) == []


def test_save_artifact_rejects_a_missing_source(storage, tmp_path):
    object_id = create_object(storage)
    with storage.update_transaction(object_id) as transaction:
        with pytest.raises(FileNotFoundError):
            transaction.save_artifact("mesh", tmp_path / "absent.ply", max_bytes=64)


@pytest.mark.parametrize("relative", [
    "../profile.json",
    "objects/00000000-0000-0000-0000-000000000000/geometry/point_cloud-11111111-1111-1111-1111-111111111111.ply",
    "objects/{object_id}/masks/point_cloud-11111111-1111-1111-1111-111111111111.ply",
    "objects/{object_id}/geometry/point_cloud-not-a-uuid.ply",
    "objects/{object_id}/geometry/unknown-11111111-1111-1111-1111-111111111111.ply",
])
def test_resolve_artifact_path_rejects_foreign_paths(storage, tmp_path, relative):
    object_id = create_object(storage)
    with pytest.raises(ValueError):
        resolve_artifact_path(storage.data_dir, object_id, relative.format(object_id=object_id), max_bytes=64)


def test_resolve_artifact_path_rejects_a_missing_file(storage):
    object_id = create_object(storage)
    relative = f"objects/{object_id}/geometry/mesh-11111111-1111-1111-1111-111111111111.ply"
    with pytest.raises(FileNotFoundError):
        resolve_artifact_path(storage.data_dir, object_id, relative, max_bytes=64)


def test_resolve_artifact_path_rejects_an_oversized_file(storage, tmp_path):
    object_id = create_object(storage)
    source = tmp_path / "big.ply"
    source.write_bytes(b"x" * 65)

    with storage.update_transaction(object_id) as transaction:
        relative = transaction.save_artifact("point_cloud", source, max_bytes=1024)
        transaction.save_profile(transaction.read_profile())
        transaction.commit()

    with pytest.raises(ValueError, match="超过大小上限"):
        resolve_artifact_path(storage.data_dir, object_id, relative, max_bytes=64)


def test_resolve_artifact_path_rejects_a_mismatched_kind(storage):
    object_id = create_object(storage)
    relative = f"objects/{object_id}/geometry/cameras-11111111-1111-1111-1111-111111111111.bin"
    transaction = storage.update_transaction(object_id)
    with transaction as active:
        source = storage.data_dir / "camera.bin"
        source.write_bytes(b"bin")
        relative = active.save_artifact("cameras", source, max_bytes=64)
        active.save_profile(active.read_profile())
        active.commit()

    with pytest.raises(ValueError, match="预期不一致"):
        resolve_artifact_path(storage.data_dir, object_id, relative, max_bytes=64, expected_kind="mesh")
