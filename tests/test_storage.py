"""Storage failures must never publish half of an object collection."""

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.object_profile import ObjectProfile
from app.services.image_collection import ImageCollectionService, ImageInput
from app.storage import local_storage
from app.storage.local_storage import LocalStorage


def assert_rolled_back(data_dir):
    for name in ("objects", ".staging"):
        directory = data_dir / name
        assert not directory.exists() or not list(directory.iterdir())


def test_partial_image_write_failure_rolls_back_and_keeps_input_open(settings, png_bytes, monkeypatch):
    original_open = Path.open

    @contextmanager
    def failing_writer(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as handle:
            class PartialWriter:
                def write(self, chunk):
                    handle.write(chunk[:8])
                    raise OSError("simulated disk write failure")

            yield PartialWriter()

    def patched_open(path, *args, **kwargs):
        if path.suffix == ".upload" and args and args[0] == "xb":
            return failing_writer(path, *args, **kwargs)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", patched_open)
    stream = BytesIO(png_bytes)
    with pytest.raises(OSError, match="simulated disk write failure"):
        ImageCollectionService(settings).collect("写盘失败", [ImageInput("image.png", stream)])
    assert not stream.closed
    assert_rolled_back(settings.data_dir)


def test_image_rename_failure_rolls_back(settings, png_bytes, monkeypatch):
    original_rename = Path.rename

    def failing_rename(path, target):
        if path.suffix == ".upload":
            raise OSError("simulated image rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    stream = BytesIO(png_bytes)
    with pytest.raises(OSError, match="simulated image rename failure"):
        ImageCollectionService(settings).collect("改名失败", [ImageInput("image.png", stream)])
    assert not stream.closed
    assert_rolled_back(settings.data_dir)


@pytest.mark.parametrize("phase", ["profile", "commit"])
def test_atomic_replace_failure_rolls_back_completed_staging(settings, png_bytes, monkeypatch, phase):
    original_replace = local_storage.os.replace
    saw_failure = []

    def failing_replace(source, destination):
        source, destination = Path(source), Path(destination)
        profile_write = destination.name == "profile.json"
        object_publish = destination.parent == settings.data_dir / "objects"
        if (phase == "profile" and profile_write) or (phase == "commit" and object_publish):
            # The failure happens after bytes have actually reached staging.
            assert source.exists()
            if object_publish:
                assert (source / "profile.json").is_file()
                assert list((source / "images").iterdir())
            saw_failure.append(True)
            raise OSError(f"simulated {phase} replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(local_storage.os, "replace", failing_replace)
    stream = BytesIO(png_bytes)
    with pytest.raises(OSError, match=f"simulated {phase} replace failure"):
        ImageCollectionService(settings).collect("原子保存失败", [ImageInput("image.png", stream)])
    assert saw_failure
    assert not stream.closed
    assert_rolled_back(settings.data_dir)


def test_commit_collision_preserves_existing_object_and_removes_only_new_staging(settings, png_bytes):
    storage = LocalStorage(settings.data_dir)
    object_id = uuid4()
    destination = storage.object_dir(object_id)
    destination.mkdir(parents=True)
    existing = destination / "profile.json"
    existing.write_bytes(b"original object must survive")

    with pytest.raises(FileExistsError):
        with storage.transaction(object_id) as transaction:
            temporary, _, _ = transaction.stage_image(BytesIO(png_bytes), uuid4(), len(png_bytes))
            transaction.finalize_image(temporary, "PNG")
            transaction.save_profile(ObjectProfile(object_id=object_id, name="重复标识"))
            transaction.commit()

    assert existing.read_bytes() == b"original object must survive"
    assert list(destination.iterdir()) == [existing]
    assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("image_format", ["BMP", "../png"])
def test_finalize_rejects_non_whitelisted_format_without_moving_file(settings, png_bytes, image_format):
    storage = LocalStorage(settings.data_dir)
    with storage.transaction(uuid4()) as transaction:
        temporary, _, _ = transaction.stage_image(BytesIO(png_bytes), uuid4(), len(png_bytes))
        with pytest.raises(ValueError):
            transaction.finalize_image(temporary, image_format)
        assert temporary.read_bytes() == png_bytes
    assert_rolled_back(settings.data_dir)


def test_finalize_cannot_move_a_file_outside_its_transaction(settings, png_bytes):
    storage = LocalStorage(settings.data_dir)
    outside = settings.data_dir.parent / "caller-owned.upload"
    outside.write_bytes(png_bytes)
    with storage.transaction(uuid4()) as transaction:
        with pytest.raises(ValueError):
            transaction.finalize_image(outside, "PNG")
    assert outside.read_bytes() == png_bytes
    assert not outside.with_suffix(".png").exists()
    assert_rolled_back(settings.data_dir)


def test_finalize_collision_preserves_existing_image(settings, png_bytes):
    storage = LocalStorage(settings.data_dir)
    with storage.transaction(uuid4()) as transaction:
        temporary, _, _ = transaction.stage_image(BytesIO(png_bytes), uuid4(), len(png_bytes))
        destination = temporary.with_suffix(".png")
        destination.write_bytes(b"existing image must survive")
        with pytest.raises(FileExistsError):
            transaction.finalize_image(temporary, "PNG")
        assert temporary.read_bytes() == png_bytes
        assert destination.read_bytes() == b"existing image must survive"
    assert_rolled_back(settings.data_dir)
