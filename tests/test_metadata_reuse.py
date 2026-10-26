"""Inspection reuse must retain real-file validation and module isolation."""

from dataclasses import FrozenInstanceError
from io import BytesIO
import os
from uuid import uuid4

import pytest
from PIL import Image

from app.models.image_asset import ImageAsset
from app.models.processing_result import ProcessingError, ProcessingResult
from app.modules import image_metadata
from app.modules.base import BaseProcessor
from app.modules.image_metadata import ImageMetadataProcessor, InspectedImage, InvalidImageError
from app.services.image_collection import ImageCollectionService, ImageInput
from app.services.pipeline import Pipeline
from app.services.profile_assembler import ProfileAssembler
from conftest import image_bytes


def make_profile(root, contents):
    object_id = uuid4()
    assets = []
    for content in contents:
        asset_id = uuid4()
        relative = f"objects/{object_id}/images/{asset_id}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        assets.append(ImageAsset(
            asset_id=asset_id, filename="photo.png", storage_path=relative,
            image_format="PNG", mime_type="image/png", width=32, height=24,
            size_bytes=len(content),
        ))
    return ProfileAssembler().create_profile("测试物体", assets, object_id)


def record_inspections(monkeypatch):
    calls = []
    original = image_metadata.inspect_image

    def inspect(path, max_pixels=40_000_000):
        calls.append((path, max_pixels))
        return original(path, max_pixels)

    monkeypatch.setattr(image_metadata, "inspect_image", inspect)
    return calls


def test_receipt_reuses_one_full_inspection_after_extension_rename(tmp_path, monkeypatch):
    profile = make_profile(tmp_path, [image_bytes()])
    asset = profile.images[0]
    final_path = tmp_path / asset.storage_path
    upload_path = final_path.with_suffix(".upload")
    final_path.rename(upload_path)
    calls = record_inspections(monkeypatch)
    inspection = InspectedImage(upload_path)
    upload_path.rename(final_path)

    completed = Pipeline([ImageMetadataProcessor()]).run(
        profile, tmp_path, inspected_images={asset.asset_id: inspection},
    )

    assert len(calls) == 1
    assert completed.processing_results[0].status == "success"
    assert completed.processing_results[0].data["images"][0] == {
        "asset_id": str(asset.asset_id), **inspection.metadata,
    }


def test_receipt_is_immutable_and_cannot_be_built_from_metadata(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(image_bytes())
    inspection = InspectedImage(path)
    extracted = inspection.metadata
    extracted["width"] = 999

    assert inspection.metadata["width"] == 32
    with pytest.raises(FrozenInstanceError):
        inspection._max_pixels = 1
    with pytest.raises(TypeError):
        InspectedImage(metadata=extracted)


def test_receipt_construction_still_rejects_invalid_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"broken")
    with pytest.raises(InvalidImageError):
        InspectedImage(path)


def test_file_changed_during_inspection_does_not_get_a_receipt(tmp_path, monkeypatch):
    path = tmp_path / "photo.png"
    path.write_bytes(image_bytes())
    original = image_metadata.inspect_image

    def inspect_then_change(source, max_pixels):
        metadata = original(source, max_pixels)
        source.write_bytes(b"changed after decoding")
        return metadata

    monkeypatch.setattr(image_metadata, "inspect_image", inspect_then_change)
    with pytest.raises(InvalidImageError, match="验证期间发生变化"):
        InspectedImage(path)


@pytest.mark.parametrize("plain_dict", [False, True])
def test_pipeline_reads_real_file_instead_of_trusting_asset_fields(tmp_path, monkeypatch, plain_dict):
    profile = make_profile(tmp_path, [image_bytes()])
    asset = profile.images[0]
    asset.width, asset.height, asset.size_bytes = 1, 1, 1
    asset.image_format, asset.mime_type = "JPEG", "image/jpeg"
    calls = record_inspections(monkeypatch)
    untrusted = {asset.asset_id: {"width": 1, "image_format": "JPEG"}} if plain_dict else None

    completed = Pipeline([ImageMetadataProcessor()]).run(
        profile, tmp_path, inspected_images=untrusted,
    )

    assert len(calls) == 1
    actual = completed.images[0]
    assert (actual.width, actual.height, actual.image_format, actual.mime_type) == (
        32, 24, "PNG", "image/png",
    )
    assert actual.size_bytes == len(image_bytes())


@pytest.mark.parametrize("plain_dict", [False, True])
def test_pipeline_cannot_skip_full_validation_with_forged_metadata(tmp_path, monkeypatch, plain_dict):
    profile = make_profile(tmp_path, [b"broken"])
    asset = profile.images[0]
    calls = record_inspections(monkeypatch)
    untrusted = {asset.asset_id: {"width": 32, "height": 24}} if plain_dict else None

    completed = Pipeline([ImageMetadataProcessor()]).run(
        profile, tmp_path, inspected_images=untrusted,
    )

    assert len(calls) == 1
    assert completed.processing_results[0].status == "failed"


def test_pipeline_without_receipt_rejects_readable_header_with_damaged_pixels(tmp_path):
    truncated = image_bytes("JPEG", size=(64, 64))[:-50]
    with Image.open(BytesIO(truncated)) as header:
        assert header.size == (64, 64)
    profile = make_profile(tmp_path, [truncated])

    completed = Pipeline([ImageMetadataProcessor()]).run(profile, tmp_path)

    assert completed.processing_results[0].status == "failed"


@pytest.mark.parametrize("change", ["replace", "modify", "remove", "rename"])
def test_changed_or_relocated_file_falls_back_to_inspection(tmp_path, monkeypatch, change):
    profile = make_profile(tmp_path, [image_bytes()])
    asset = profile.images[0]
    path = tmp_path / asset.storage_path
    calls = record_inspections(monkeypatch)
    inspection = InspectedImage(path)
    if change == "replace":
        replacement = path.with_name("replacement.png")
        replacement.write_bytes(image_bytes())
        # Identical bytes/size/mtime still must not reuse a different file identity.
        original_stat = path.stat()
        os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        replacement.replace(path)
    elif change == "modify":
        path.write_bytes(b"broken")
    elif change == "remove":
        path.unlink()
    else:
        renamed = path.with_name("different-name.png")
        path.rename(renamed)
        asset.storage_path = renamed.relative_to(tmp_path).as_posix()

    completed = Pipeline([ImageMetadataProcessor()]).run(
        profile, tmp_path, inspected_images={asset.asset_id: inspection},
    )

    assert len(calls) == 2
    expected = "failed" if change in {"modify", "remove"} else "success"
    assert completed.processing_results[0].status == expected


@pytest.mark.parametrize("max_pixels, expected", [(900, "success"), (500, "failed")])
def test_tighter_pixel_limit_requires_fresh_inspection(tmp_path, monkeypatch, max_pixels, expected):
    profile = make_profile(tmp_path, [image_bytes()])
    asset = profile.images[0]
    calls = record_inspections(monkeypatch)
    inspection = InspectedImage(tmp_path / asset.storage_path, max_pixels=1_000)

    completed = Pipeline([ImageMetadataProcessor()]).run(
        profile, tmp_path, max_image_pixels=max_pixels,
        inspected_images={asset.asset_id: inspection},
    )

    assert len(calls) == 2
    assert completed.processing_results[0].status == expected


@pytest.mark.parametrize("contents, expected", [
    ([image_bytes(), b"broken"], "partial"),
    ([b"broken", b"also broken"], "failed"),
])
def test_metadata_partial_and_total_failure_are_retained(tmp_path, contents, expected):
    profile = make_profile(tmp_path, contents)
    completed = Pipeline([ImageMetadataProcessor()]).run(profile, tmp_path)
    result = completed.processing_results[0]

    assert result.status == expected
    assert len(result.errors) == (1 if expected == "partial" else 2)
    assert completed.status == "completed_with_warnings"
    assert len(completed.images) == 2


def test_module_snapshot_mutations_do_not_reach_later_modules(tmp_path):
    profile = make_profile(tmp_path, [image_bytes()])

    class SeedProcessor(BaseProcessor):
        name = "seed"

        def process(self, context):
            return ProcessingResult(module_name=self.name, status="success", data={"nested": {"value": "original"}})

    class MutatingProcessor(BaseProcessor):
        name = "mutating"

        def process(self, context):
            context.images[0].width = -1
            context.profile.name = "污染"
            context.previous_results[0].data["nested"]["value"] = "污染"
            with pytest.raises(TypeError):
                context.inspected_images[context.images[0].asset_id] = {}
            return ProcessingResult(module_name=self.name, status="success")

    class ObserverProcessor(BaseProcessor):
        name = "observer"

        def process(self, context):
            assert context.images[0].width == 32
            assert context.profile.name == "测试物体"
            assert context.previous_results[0].data["nested"]["value"] == "original"
            return ProcessingResult(module_name=self.name, status="success")

    completed = Pipeline([SeedProcessor(), MutatingProcessor(), ObserverProcessor()]).run(profile, tmp_path)

    assert all(result.status == "success" for result in completed.processing_results)
    assert profile.name == "测试物体"
    assert profile.images[0].width == 32
    assert completed.processing_results[0].data["nested"]["value"] == "original"


def test_collection_fully_inspects_each_image_only_once(settings, png_bytes, monkeypatch):
    calls = record_inspections(monkeypatch)
    service = ImageCollectionService(settings)

    profile = service.collect("双图采集", [
        ImageInput("front.png", BytesIO(png_bytes)),
        ImageInput("back.png", BytesIO(png_bytes)),
    ])

    assert len(calls) == 2
    assert {path.stem for path, _ in calls} == {str(asset.asset_id) for asset in profile.images}
    result = profile.processing_results[0]
    assert result.module_name == "image_metadata"
    assert result.status == "success"
    assert len(result.data["images"]) == 2
    assert service.get_profile(profile.object_id) == profile
    assert list((settings.data_dir / ".staging").iterdir()) == []


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_collection_preserves_archive_when_processing_is_partial_or_all_failed(settings, png_bytes, status):
    class OutcomeProcessor(BaseProcessor):
        name = "test_outcome"

        def process(self, context):
            if status == "failed":
                raise RuntimeError("simulated processing failure")
            return ProcessingResult(
                module_name=self.name,
                status="partial",
                data={"accepted_asset_ids": [str(context.images[0].asset_id)]},
                errors=[ProcessingError(
                    code="simulated_image_failure", message="一张图片的处理结果不可用。",
                    asset_id=context.images[1].asset_id,
                )],
            )

    service = ImageCollectionService(settings, processors=[OutcomeProcessor()])
    profile = service.collect("保存有效输入", [
        ImageInput("front.png", BytesIO(png_bytes)),
        ImageInput("back.png", BytesIO(png_bytes)),
    ])

    assert profile.status == "completed_with_warnings"
    assert [result.status for result in profile.processing_results] == [status]
    assert profile.processing_results[0].errors
    assert service.get_profile(profile.object_id) == profile
    assert len(profile.images) == 2
    assert all((settings.data_dir / image.storage_path).read_bytes() == png_bytes for image in profile.images)
    assert (settings.data_dir / "objects" / str(profile.object_id) / "profile.json").is_file()
    assert list((settings.data_dir / ".staging").iterdir()) == []
    assert not list(settings.data_dir.rglob("*.upload"))
