from uuid import uuid4

import pytest

from app.models.image_asset import ImageAsset
from app.models.processing_result import ProcessingResult
from app.modules.base import BaseProcessor
from app.modules.image_metadata import ImageMetadataProcessor
from app.services.pipeline import Pipeline
from app.services.profile_assembler import ProfileAssembler
from conftest import image_bytes


@pytest.fixture
def initial_profile(tmp_path):
    object_id, asset_id = uuid4(), uuid4()
    content = image_bytes()
    relative_path = f"objects/{object_id}/images/{asset_id}.png"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    asset = ImageAsset(
        asset_id=asset_id,
        filename="front.png",
        storage_path=relative_path,
        mime_type="image/png",
        width=32,
        height=24,
        size_bytes=len(content),
    )
    return ProfileAssembler().create_profile("测试物体", [asset], object_id=object_id)


class CategoryProcessor(BaseProcessor):
    name = "test_category"

    def process(self, context):
        assert isinstance(context.images, tuple)
        assert context.resolve_asset(context.images[0]).is_file()
        return ProcessingResult(module_name=self.name, status="success", data={"category": "测试类别"})


class ExplodingProcessor(BaseProcessor):
    name = "exploding"

    def process(self, context):
        raise RuntimeError("PRIVATE_INTERNAL_PATH_C:\\private\\model")


class InvalidMetadataProcessor(BaseProcessor):
    name = "image_metadata"

    def process(self, context):
        return ProcessingResult(module_name=self.name, status="success", data={"images": [{"asset_id": "not-a-uuid", "width": "wrong"}]})


def test_real_metadata_processor_reads_image(initial_profile, tmp_path):
    profile = Pipeline([ImageMetadataProcessor()]).run(initial_profile, tmp_path)
    result = profile.processing_results[0]
    assert result.module_name == "image_metadata"
    assert result.status == "success"
    metadata = result.data["images"][0]
    assert metadata["image_format"] == "PNG"
    assert metadata["asset_id"] == str(initial_profile.images[0].asset_id)
    assert (metadata["width"], metadata["height"]) == (32, 24)
    assert metadata["size_bytes"] == initial_profile.images[0].size_bytes
    assert initial_profile.processing_results == []


def test_registered_adapter_merges_typed_profile_and_next_processor_sees_result(initial_profile, tmp_path):
    assembler = ProfileAssembler()

    def apply_category(profile, result):
        profile.category = result.data["category"]
        return profile

    assembler.register_adapter("test_category", apply_category)

    class LaterProcessor(BaseProcessor):
        name = "later"

        def process(self, context):
            assert context.profile.category == "测试类别"
            assert context.previous_results[0].module_name == "test_category"
            return ProcessingResult(module_name=self.name, status="skipped", data={})

    profile = Pipeline([CategoryProcessor(), LaterProcessor()], assembler=assembler).run(initial_profile, tmp_path)
    assert profile.category == "测试类别"
    assert "category" not in profile.missing_information
    assert [result.module_name for result in profile.processing_results] == ["test_category", "later"]
    assert profile.processing_results[1].status == "skipped"


def test_module_exception_is_tracked_and_does_not_discard_other_results(initial_profile, tmp_path):
    profile = Pipeline([ExplodingProcessor(), ImageMetadataProcessor()]).run(initial_profile, tmp_path)
    assert profile.object_id == initial_profile.object_id
    assert len(profile.images) == 1
    assert profile.status == "completed_with_warnings"
    failed, succeeded = profile.processing_results
    assert failed.status == "failed"
    assert failed.errors
    assert failed.errors[0].code
    assert failed.errors[0].error_id is not None
    assert succeeded.status == "success"
    assert "PRIVATE_INTERNAL_PATH" not in profile.model_dump_json()


def test_invalid_module_payload_is_isolated(initial_profile, tmp_path):
    profile = Pipeline([InvalidMetadataProcessor(), CategoryProcessor()]).run(initial_profile, tmp_path)
    assert profile.processing_results[0].status == "failed"
    assert profile.processing_results[0].errors
    assert profile.processing_results[1].status == "success"
    assert (profile.images[0].width, profile.images[0].height) == (32, 24)


def test_adapter_cannot_replace_object_identity(initial_profile, tmp_path):
    assembler = ProfileAssembler()

    def replace_identity(profile, result):
        profile.object_id = uuid4()
        return profile

    assembler.register_adapter("test_category", replace_identity)
    profile = Pipeline([CategoryProcessor(), ImageMetadataProcessor()], assembler=assembler).run(initial_profile, tmp_path)
    assert profile.object_id == initial_profile.object_id
    assert profile.processing_results[0].status == "failed"
    assert profile.processing_results[1].status == "success"


def test_module_failure_still_creates_downloadable_archive(client_factory):
    client, settings = client_factory(processors=[ExplodingProcessor(), ImageMetadataProcessor()])
    response = client.post(
        "/api/v1/objects", data={"name": "保留部分结果"},
        files=[("images", ("front.png", image_bytes(), "image/png"))],
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["status"] == "completed_with_warnings"
    assert [result["status"] for result in created["profile"]["processing_results"]] == ["failed", "success"]
    assert "PRIVATE_INTERNAL_PATH" not in response.text
    assert client.get(f"/api/v1/objects/{created['object_id']}/profile.json").status_code == 200
    assert (settings.data_dir / "objects" / created["object_id"] / "profile.json").is_file()
