"""Mask artifacts share object transactions and reject invalid ownership or bytes."""

from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw
import pytest

from app.models.image_asset import ImageAsset
from app.models.processing_result import ProcessingResult
from app.models.segmentation import NormalizedBox, SegmentationData, SegmentationImage
from app.modules.base import BaseProcessor
from app.services.pipeline import Pipeline
from app.services.profile_assembler import ProfileAssembler
from app.storage import local_storage
from app.storage.local_storage import LocalStorage, resolve_mask_path, storage_io_path


def staged_profile(transaction, png_bytes):
    asset_id = uuid4()
    temporary, size, checksum = transaction.stage_image(BytesIO(png_bytes), asset_id, len(png_bytes))
    relative = transaction.finalize_image(temporary, "PNG")
    asset = ImageAsset(
        asset_id=asset_id, filename="photo.png", storage_path=relative, width=32, height=24,
        image_format="PNG", mime_type="image/png", size_bytes=size, checksum=checksum,
    )
    return ProfileAssembler().create_profile("待分割物体", [asset], transaction.object_id)


def binary_mask(size=(32, 24)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((4, 5, 11, 13), fill=255)
    return mask


def mask_result(asset, mask_path):
    image = SegmentationImage(
        asset_id=asset.asset_id, filename=asset.filename, status="success",
        prompt_type="box", prompt_box=NormalizedBox(x0=0.1, y0=0.1, x1=0.8, y1=0.8),
        width=32, height=24, mask_path=mask_path, bbox=(4, 5, 12, 14),
        area_ratio=72 / (32 * 24), confidence=0.9,
    )
    payload = SegmentationData(
        input_count=1, success_count=1, review_count=0, failed_count=0, skipped_count=0, images=[image],
    )
    return ProcessingResult(module_name="object_segmentation", status="success", data=payload.model_dump(mode="json"))


def assert_no_publish(data_dir):
    for name in ("objects", ".staging"):
        directory = data_dir / name
        assert not directory.exists() or not list(directory.iterdir())


def test_mask_is_staged_then_committed_with_the_profile(settings, png_bytes):
    storage = LocalStorage(settings.data_dir)
    object_id = uuid4()
    with storage.transaction(object_id) as transaction:
        profile = staged_profile(transaction, png_bytes)
        asset = profile.images[0]
        with binary_mask() as mask:
            relative = transaction.save_mask(asset.asset_id, mask)
        assert relative.startswith(f"objects/{object_id}/masks/{asset.asset_id}_")
        assert not (settings.data_dir / relative).exists()
        assert resolve_mask_path(transaction.data_dir, object_id, asset.asset_id, relative, (32, 24)).is_file()
        profile.processing_results.append(mask_result(asset, relative))
        transaction.save_profile(profile)
        transaction.commit()

    path = resolve_mask_path(settings.data_dir, object_id, asset.asset_id, relative, (32, 24))
    with Image.open(path) as image:
        assert image.mode == "L"
        assert image.getbbox() == (4, 5, 12, 14)
        assert {value for _, value in image.getcolors()} == {0, 255}
    persisted, _ = storage.read_profile(object_id)
    assert persisted.processing_results[0].data["images"][0]["mask_path"] == relative
    assert (settings.data_dir / asset.storage_path).read_bytes() == png_bytes
    assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("invalid", ["color", "mid_gray", "empty", "wrong_size"])
def test_save_mask_rejects_invalid_content_before_creating_artifacts(settings, png_bytes, invalid):
    storage = LocalStorage(settings.data_dir)
    with storage.transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        mask = {"color": Image.new("RGB", (32, 24), "white"),
                "mid_gray": Image.new("L", (32, 24), 128),
                "empty": Image.new("L", (32, 24), 0),
                "wrong_size": binary_mask((40, 30))}[invalid]
        with mask, pytest.raises(ValueError):
            transaction.save_mask(profile.images[0].asset_id, mask)
        assert not (transaction.object_dir / "masks").exists()
    assert_no_publish(settings.data_dir)


def test_save_mask_rejects_unknown_asset(settings, png_bytes):
    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        staged_profile(transaction, png_bytes)
        with binary_mask() as mask, pytest.raises(ValueError):
            transaction.save_mask(uuid4(), mask)
    assert_no_publish(settings.data_dir)


def test_save_mask_uses_exif_oriented_dimensions(settings):
    stream = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (32, 24), "gray").save(stream, format="JPEG", exif=exif)
    content = stream.getvalue()
    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        asset_id = uuid4()
        temporary, _, _ = transaction.stage_image(BytesIO(content), asset_id, len(content))
        transaction.finalize_image(temporary, "JPEG")
        with binary_mask((24, 32)) as mask:
            relative = transaction.save_mask(asset_id, mask)
        assert resolve_mask_path(transaction.data_dir, transaction.object_id, asset_id, relative, (24, 32)).is_file()
        with binary_mask() as wrong_orientation, pytest.raises(ValueError):
            transaction.save_mask(asset_id, wrong_orientation)


@pytest.mark.parametrize("phase", ["write", "replace"])
def test_mask_write_or_replace_failure_leaves_no_partial_mask(settings, png_bytes, monkeypatch, phase):
    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        if phase == "write":
            def broken_save(self, handle, *args, **kwargs):
                handle.write(b"partial PNG")
                raise OSError("simulated mask write failure")
            monkeypatch.setattr(Image.Image, "save", broken_save)
        else:
            def broken_replace(source, destination):
                raise OSError("simulated mask replace failure")
            monkeypatch.setattr(local_storage.os, "replace", broken_replace)
        with binary_mask() as mask, pytest.raises(OSError, match="simulated mask"):
            transaction.save_mask(profile.images[0].asset_id, mask)
        assert list((transaction.object_dir / "masks").iterdir()) == []
    assert_no_publish(settings.data_dir)


def test_failed_temporary_cleanup_cannot_publish_a_partial_mask(settings, png_bytes, monkeypatch):
    original_unlink = Path.unlink

    class WritesBrokenMask(BaseProcessor):
        name = "object_segmentation"

        def process(self, context):
            with binary_mask() as mask:
                context.artifact_store.save_mask(context.images[0].asset_id, mask)
            raise AssertionError("The storage failure should be isolated first")

    def broken_save(self, handle, *args, **kwargs):
        handle.write(b"partial PNG")
        raise OSError("simulated mask write failure")

    def broken_cleanup(path, *args, **kwargs):
        if path.suffix == ".tmp" and path.parent.name == "masks":
            raise PermissionError("simulated temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with pytest.raises(PermissionError, match="temporary cleanup"):
        with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
            profile = staged_profile(transaction, png_bytes)
            monkeypatch.setattr(Image.Image, "save", broken_save)
            monkeypatch.setattr(Path, "unlink", broken_cleanup)
            completed = Pipeline([WritesBrokenMask()]).run(profile, transaction.data_dir, artifact_store=transaction)
            assert completed.processing_results[0].status == "failed"
            assert list((transaction.object_dir / "masks").glob("*.tmp"))
            transaction.save_profile(completed)
            transaction.commit()
    assert_no_publish(settings.data_dir)


@pytest.mark.parametrize("phase", ["profile", "commit"])
def test_storage_failure_rolls_back_masks_and_originals(settings, png_bytes, monkeypatch, phase):
    storage = LocalStorage(settings.data_dir)
    original_replace = local_storage.os.replace

    def fail(source, destination):
        destination = Path(destination)
        if (phase == "profile" and destination.name == "profile.json") or (
            phase == "commit" and destination.parent == settings.data_dir / "objects"
        ):
            raise OSError("simulated archive failure")
        return original_replace(source, destination)

    with pytest.raises(OSError, match="simulated archive"):
        with storage.transaction(uuid4()) as transaction:
            profile = staged_profile(transaction, png_bytes)
            with binary_mask() as mask:
                relative = transaction.save_mask(profile.images[0].asset_id, mask)
            profile.processing_results.append(mask_result(profile.images[0], relative))
            monkeypatch.setattr(local_storage.os, "replace", fail)
            transaction.save_profile(profile)
            transaction.commit()
    assert_no_publish(settings.data_dir)


@pytest.mark.parametrize("failure", ["module", "adapter"])
def test_saved_masks_are_removed_if_the_module_or_adapter_fails(settings, png_bytes, failure):
    class WritesMask(BaseProcessor):
        name = "object_segmentation"

        def process(self, context):
            with binary_mask() as mask:
                relative = context.artifact_store.save_mask(context.images[0].asset_id, mask)
            if failure == "module":
                raise RuntimeError("simulated failure after mask save")
            return mask_result(context.images[0], relative)

    assembler = ProfileAssembler()
    if failure == "adapter":
        def reject(profile, result):
            raise ValueError("simulated adapter rejection")
        assembler.register_adapter("object_segmentation", reject)
    storage = LocalStorage(settings.data_dir)
    with storage.transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        completed = Pipeline([WritesMask()], assembler).run(profile, transaction.data_dir, artifact_store=transaction)
        assert completed.processing_results[0].status == "failed"
        assert completed.processing_results[0].data == {}
        assert list((transaction.object_dir / "masks").iterdir())
        transaction.save_profile(completed)
        assert list((transaction.object_dir / "masks").iterdir()) == []
        transaction.commit()
    assert (settings.data_dir / profile.images[0].storage_path).read_bytes() == png_bytes


def test_pipeline_rejects_an_absent_mask_but_keeps_the_archive(settings, png_bytes):
    class MissingMask(BaseProcessor):
        name = "object_segmentation"

        def process(self, context):
            asset = context.images[0]
            return mask_result(asset, f"objects/{context.profile.object_id}/masks/{asset.asset_id}_{uuid4()}.png")

    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        completed = Pipeline([MissingMask()]).run(profile, transaction.data_dir, artifact_store=transaction)
        assert completed.processing_results[0].status == "failed"
        assert completed.images == profile.images
        transaction.save_profile(completed)
        transaction.commit()


@pytest.mark.parametrize("fault", ["object", "asset", "extension", "traversal", "missing", "pixels"])
def test_mask_resolver_rejects_invalid_ownership_paths_and_limits(settings, png_bytes, fault):
    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        asset_id = profile.images[0].asset_id
        with binary_mask() as mask:
            relative = transaction.save_mask(asset_id, mask)
        target = relative
        if fault == "object":
            target = relative.replace(str(transaction.object_id), str(uuid4()))
        elif fault == "asset":
            target = relative.replace(str(asset_id), str(uuid4()))
        elif fault == "extension":
            target = relative.removesuffix(".png") + ".jpg"
        elif fault == "traversal":
            target = "../outside.png"
        elif fault == "missing":
            storage_io_path(transaction.data_dir / relative).unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            resolve_mask_path(transaction.data_dir, transaction.object_id, asset_id, target,
                              max_pixels=767 if fault == "pixels" else 40_000_000)


@pytest.mark.parametrize("content", ["empty", "mid_gray", "wrong_size", "color"])
def test_mask_resolver_checks_actual_png_content(settings, png_bytes, content):
    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        asset_id = profile.images[0].asset_id
        with binary_mask() as mask:
            relative = transaction.save_mask(asset_id, mask)
        replacements = {"empty": Image.new("L", (32, 24), 0), "mid_gray": Image.new("L", (32, 24), 128),
                        "wrong_size": binary_mask((40, 30)), "color": Image.new("RGB", (32, 24), "white")}
        with replacements[content] as replacement:
            replacement.save(storage_io_path(transaction.data_dir / relative), format="PNG")
        with pytest.raises(ValueError):
            resolve_mask_path(transaction.data_dir, transaction.object_id, asset_id, relative, (32, 24))


@pytest.mark.parametrize("geometry", ["bbox", "area"])
def test_mask_resolver_rejects_forged_geometry(settings, png_bytes, geometry):
    with LocalStorage(settings.data_dir).transaction(uuid4()) as transaction:
        profile = staged_profile(transaction, png_bytes)
        asset_id = profile.images[0].asset_id
        with binary_mask() as mask:
            relative = transaction.save_mask(asset_id, mask)
        with pytest.raises(ValueError):
            resolve_mask_path(
                transaction.data_dir, transaction.object_id, asset_id, relative,
                expected_bbox=(0, 0, 10, 10) if geometry == "bbox" else (4, 5, 12, 14),
                expected_area_ratio=0.9 if geometry == "area" else 72 / 768,
            )


def test_pipeline_passes_readonly_boxes_and_rejects_another_transactions_store(settings, png_bytes):
    box = NormalizedBox(x0=0.1, y0=0.2, x1=0.8, y1=0.9)

    class InspectContext(BaseProcessor):
        name = "context_test"

        def process(self, context):
            assert context.segmentation_boxes[context.images[0].asset_id] == box
            with pytest.raises(TypeError):
                context.segmentation_boxes[context.images[0].asset_id] = box
            assert context.artifact_store is transaction
            return ProcessingResult(module_name=self.name, status="success")

    storage = LocalStorage(settings.data_dir)
    with storage.transaction(uuid4()) as transaction, storage.transaction(uuid4()) as unrelated:
        profile = staged_profile(transaction, png_bytes)
        completed = Pipeline([InspectContext()]).run(
            profile, transaction.data_dir, artifact_store=transaction,
            segmentation_boxes={profile.images[0].asset_id: box},
        )
        assert completed.processing_results[0].status == "success"
        with pytest.raises(ValueError, match="当前处理事务"):
            Pipeline([]).run(profile, transaction.data_dir, artifact_store=unrelated)
