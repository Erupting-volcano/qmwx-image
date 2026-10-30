"""Registry/configuration tests. No real model or paid provider requests occur."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from app.config import Settings
from app.modules.segmentation_backend import (
    BackendInitializationError, BoxPrompt, NotImplementedBackendError, SegmentationBackend,
    SegmentationCapabilities, SegmentationRequest, SegmentationUnavailable, TextPrompt,
    UnknownBackendError, UnsupportedPromptError,
)
from app.modules.segmentation_factory import (
    BAIDU_CAPABILITIES, SAM2_CAPABILITIES, BackendRegistration, SEGMENTATION_REGISTRY,
    SegmentationRegistry, create_backend, describe_backend, get_backend_registration,
)


def config(tmp_path, **kwargs):
    values = dict(data_dir=tmp_path, segmentation_backend="baidu", segmentation_enabled=False,
                  segmentation_api_key="", segmentation_secret_key="")
    values.update(kwargs)
    return Settings(**values)


def test_default_backend_is_baidu_and_can_be_selected_by_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("DIGITIZER_SEGMENTATION_BACKEND", raising=False)
    assert Settings(data_dir=tmp_path).segmentation_backend == "baidu"
    monkeypatch.setenv("DIGITIZER_SEGMENTATION_BACKEND", "sam2.1_hiera_tiny")
    assert Settings(data_dir=tmp_path).segmentation_backend == "sam2.1_hiera_tiny"
    monkeypatch.setenv("DIGITIZER_SEGMENTATION_BACKEND", "baidu")
    assert Settings(data_dir=tmp_path).segmentation_backend == "baidu"


def test_builtins_have_distinct_deployment_readiness_and_capabilities(tmp_path):
    baidu = get_backend_registration("baidu")
    sam2 = get_backend_registration("sam2.1_hiera_tiny")
    assert baidu.deployment == "cloud" and baidu.status == "available"
    assert baidu.model_name == "baidu_intelligent_segmentation"
    assert baidu.capabilities.box_prompt and baidu.capabilities.automatic_mask
    assert not baidu.capabilities.point_prompt and not baidu.capabilities.quality_score
    assert sam2.deployment == "local" and sam2.status == "not_implemented"
    assert sam2.capabilities == SAM2_CAPABILITIES
    assert sam2.capabilities.point_prompt and sam2.capabilities.mask_prompt
    assert sam2.capabilities.multiple_masks and sam2.capabilities.iterative_refinement
    assert sam2.capabilities.quality_score and not sam2.capabilities.text_prompt
    assert len(SEGMENTATION_REGISTRY.list_models()) >= 2
    # Configuration readiness is metadata; it does not contact the service.
    assert describe_backend("baidu", config(tmp_path)).status == "unavailable"
    assert get_backend_registration("baidu").status == "available"


def test_baidu_factory_constructs_lazy_standard_adapter_without_a_request(tmp_path, monkeypatch):
    from app.modules.baidu_backend import BaiduSegmentationBackend

    monkeypatch.setattr(BaiduSegmentationBackend, "_access_token", lambda self: pytest.fail("No token call permitted"))
    instance = create_backend(config(tmp_path, segmentation_api_key="TEST_API_KEY", segmentation_secret_key="TEST_SECRET"))
    assert isinstance(instance, SegmentationBackend)
    assert instance.model_id == "baidu" and instance.capabilities == BAIDU_CAPABILITIES
    assert instance._token is None


def test_missing_baidu_credentials_are_explicit_and_never_use_sam2(tmp_path):
    with pytest.raises(SegmentationUnavailable, match="API Key"):
        create_backend(config(tmp_path, segmentation_enabled=True))


def test_selected_unimplemented_model_does_not_require_cloud_credentials(tmp_path):
    settings = config(tmp_path, segmentation_backend="sam2.1_hiera_tiny", segmentation_enabled=True)
    assert describe_backend(settings.segmentation_backend, settings).status == "not_implemented"
    with pytest.raises(NotImplementedBackendError, match="尚未实现"):
        create_backend(settings)


def test_unknown_model_registration_and_creation_have_clear_errors(tmp_path):
    with pytest.raises(UnknownBackendError, match="未注册"):
        get_backend_registration("missing")
    with pytest.raises(UnknownBackendError, match="未注册"):
        create_backend(config(tmp_path, segmentation_backend="missing"))


class ContractOnlyBackend:
    """A test adapter which is never asked to produce a mask."""
    model_id = "test_backend"
    model_name = "test_backend"
    model_version = "test"
    capabilities = SegmentationCapabilities(text_prompt=True)

    def predict(self, request):
        raise AssertionError("This test only checks construction, never inference")


def registration(builder):
    return BackendRegistration(
        model_id="test_backend", name="Test framework adapter", model_name="test_backend",
        model_version="test", description="Only metadata, no real model inference.",
        capabilities=ContractOnlyBackend.capabilities, deployment="local", status="available", builder=builder,
    )


def test_third_backend_registration_does_not_change_processor_or_builtin_factory(tmp_path):
    registry = SegmentationRegistry()
    calls = []
    model = registration(lambda settings: calls.append(settings) or ContractOnlyBackend())
    registry.register(model)
    assert calls == []
    assert registry.get("test_backend") is model
    assert registry.describe("test_backend").capabilities.text_prompt
    instance = registry.create("test_backend", config(tmp_path))
    assert isinstance(instance, ContractOnlyBackend) and len(calls) == 1
    with pytest.raises(ValueError, match="已经注册"):
        registry.register(model)
    with pytest.raises(UnknownBackendError):
        SEGMENTATION_REGISTRY.get("test_backend")


def test_initialization_errors_are_sanitized_and_not_retried_elsewhere(tmp_path):
    calls = []

    def broken(settings):
        calls.append("called")
        raise RuntimeError("PRIVATE_API_KEY PRIVATE_CHECKPOINT_PATH")

    registry = SegmentationRegistry()
    registry.register(registration(broken))
    with pytest.raises(BackendInitializationError) as caught:
        registry.create("test_backend", config(tmp_path))
    assert calls == ["called"]
    assert "PRIVATE" not in str(caught.value) and caught.value.__suppress_context__


def test_backend_identity_and_capability_must_match_registration(tmp_path):
    registry = SegmentationRegistry()
    registry.register(registration(lambda settings: object()))
    with pytest.raises(BackendInitializationError):
        registry.create("test_backend", config(tmp_path))


def test_available_registration_requires_capability_structure_and_builder():
    good = registration(lambda settings: ContractOnlyBackend())
    with pytest.raises(TypeError):
        replace(good, capabilities={"text_prompt": True})
    with pytest.raises(ValueError):
        replace(good, builder=None)


def test_public_metadata_has_no_credentials_or_private_configuration(tmp_path):
    settings = config(tmp_path, segmentation_api_key="PRIVATE_API_KEY", segmentation_secret_key="PRIVATE_SECRET",
                      sam2_checkpoint_path=tmp_path / "private-location.pt")
    metadata = describe_backend("baidu", settings).to_public_dict()
    serialized = json.dumps(metadata)
    assert metadata["status"] == "available"
    assert all(value not in serialized for value in ("PRIVATE_API_KEY", "PRIVATE_SECRET", "private-location.pt"))
    assert "builder" not in metadata and "configuration_check" not in metadata
    assert metadata["capabilities"]["automatic_mask"] is True
    assert "PRIVATE" not in repr(settings) and "private-location.pt" not in repr(settings)


def test_sam2_configuration_is_only_a_future_interface(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM2_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("SAM2_DEVICE", "cuda")
    settings = Settings(data_dir=tmp_path, segmentation_backend="sam2.1_hiera_tiny")
    assert settings.sam2_checkpoint_path == tmp_path / "missing.pt" and settings.sam2_device == "cuda"
    assert describe_backend(settings.segmentation_backend, settings).status == "not_implemented"


def test_explicit_sam2_diagnostic_never_imports_or_runs_a_model(monkeypatch, tmp_path):
    from app.modules import sam2_backend

    observed = []
    monkeypatch.setattr(sam2_backend, "find_spec", lambda name: observed.append(name) or None)
    instance = sam2_backend.Sam2Backend(tmp_path / "missing.pt", "cpu")
    diagnostic = instance.check_dependencies()
    assert observed == ["torch", "torchvision", "sam2"]
    assert diagnostic["missing_dependencies"] == ("torch", "torchvision", "sam2")
    assert diagnostic["status"] == "not_implemented" and not diagnostic["checkpoint_exists"]
    with Image.new("RGB", (20, 20)) as image:
        with pytest.raises(NotImplementedBackendError):
            instance.predict(SegmentationRequest(image, BoxPrompt((1, 1, 19, 19))))
        with pytest.raises(UnsupportedPromptError):
            instance.predict(SegmentationRequest(image, TextPrompt("牛奶盒")))


def test_sam2_remains_unimplemented_even_with_present_packages_and_checkpoint(monkeypatch, tmp_path):
    from app.modules import sam2_backend

    monkeypatch.setattr(sam2_backend, "find_spec", lambda name: object())
    checkpoint = tmp_path / "test-config-only.pt"
    checkpoint.write_text("Test placeholder, not a neural checkpoint")
    instance = sam2_backend.Sam2Backend(checkpoint, "cuda")
    diagnostic = instance.check_dependencies()
    assert diagnostic["missing_dependencies"] == () and diagnostic["checkpoint_exists"]
    assert diagnostic["status"] == "not_implemented"
    with Image.new("RGB", (20, 20)) as image, pytest.raises(NotImplementedBackendError):
        instance.predict(SegmentationRequest(image, BoxPrompt((1, 1, 19, 19))))


def test_application_startup_and_metadata_never_import_optional_model_dependencies(tmp_path):
    # A clean interpreter matters: other tests explicitly import stub/adapter modules.
    script = """
import sys
from app.config import Settings
from app.main import create_app
from app.modules.segmentation_factory import describe_backend
for selected in ('baidu', 'sam2.1_hiera_tiny', 'unknown_model'):
    settings = Settings(data_dir=sys.argv[1], segmentation_enabled=True,
                        segmentation_backend=selected, segmentation_api_key='', segmentation_secret_key='')
    create_app(settings)
    if selected != 'unknown_model': describe_backend(selected, settings)
for name in ('torch','torchvision','sam2','app.modules.sam2_backend','app.modules.baidu_backend'):
    assert name not in sys.modules, name
print('No backend imports or requests at startup')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=Path(__file__).parents[1],
                            capture_output=True, text=True, check=True)
    assert "No backend imports" in result.stdout
