"""Model/task routing tests without initializing or calling a real model service."""

from dataclasses import replace
import json
import traceback

import pytest

from app.config import Settings
from app.modules.recognition_backend import RecognitionCapabilities, RecognitionUnavailable
from app.modules.recognition_factory import (
    RECOGNITION_REGISTRY, RecognitionRegistration, RecognitionRegistry, RecognitionRouter, resolve_model_config,
)


def settings(tmp_path, **kwargs):
    values = dict(data_dir=tmp_path, recognition_enabled=False, recognition_api_key="", recognition_model="qwen3-vl-plus",
                  recognition_task_models={}, recognition_model_configs={})
    values.update(kwargs)
    return Settings(**values)


class TestBackend:
    __test__ = False
    model_id = "test_model"
    model_version = "test-only"
    capabilities = RecognitionCapabilities(image_input=True, structured_output=True)

    def analyze(self, request):
        raise AssertionError("Only framework initialization is under test")


def registration(builder=None, **kwargs):
    values = dict(model_id="test_model", provider="test-only", model_name="test-api-name", model_version="test-only",
                  capabilities=TestBackend.capabilities, deployment_type="local", builder=builder or (lambda config: TestBackend()))
    values.update(kwargs)
    return RecognitionRegistration(**values)


def test_default_config_and_registered_capabilities_match_verified_adapter_scope(tmp_path, monkeypatch):
    monkeypatch.delenv("DIGITIZER_RECOGNITION_MODEL", raising=False)
    monkeypatch.delenv("DIGITIZER_RECOGNITION_BACKEND", raising=False)
    monkeypatch.delenv("DIGITIZER_RECOGNITION_ENABLED", raising=False)
    current = Settings(data_dir=tmp_path)
    assert current.recognition_enabled is False and current.recognition_model == "qwen3-vl-plus"
    assert current.recognition_backend == "openai_compatible" and current.recognition_combine_tasks is True
    model = RECOGNITION_REGISTRY.get("qwen3-vl-plus")
    assert model.capabilities.image_input and model.capabilities.multi_image_input and model.capabilities.structured_output
    assert not any((model.capabilities.json_schema, model.capabilities.high_resolution_image,
                    model.capabilities.multi_turn, model.capabilities.video_input))


def test_model_registration_lookup_and_duplicate_protection():
    registry = RecognitionRegistry()
    item = registration()
    registry.register(item)
    assert registry.get("test_model") is item and registry.list_models() == (item,)
    with pytest.raises(ValueError, match="已注册"):
        registry.register(item)
    with pytest.raises(RecognitionUnavailable, match="未注册"):
        registry.get("unknown")


def test_router_default_and_task_override_are_lazy_and_do_not_initialize_unchosen_models(tmp_path):
    calls = []
    registry = RecognitionRegistry()
    registry.register(registration(builder=lambda config: calls.append("test") or TestBackend()))
    class Other(TestBackend):
        model_id = "other_model"
    registry.register(registration(model_id="other_model", builder=lambda config: calls.append("other") or Other()))
    router = RecognitionRouter(settings(tmp_path, recognition_model="test_model", recognition_task_models={"material": "other_model"}), registry)
    assert router.model_id_for_task("classification") == "test_model"
    assert router.model_id_for_task("material") == "other_model" and calls == []
    first = router.get_backend("classification")
    assert router.get_backend("appearance") is first and calls == ["test"]
    assert router.get_backend("material").model_id == "other_model" and calls == ["test", "other"]


def test_unknown_override_does_not_fall_back_to_configured_default(tmp_path):
    registry = RecognitionRegistry()
    registry.register(registration())
    router = RecognitionRouter(settings(tmp_path, recognition_model="test_model", recognition_task_models={"material": "unknown"}), registry)
    with pytest.raises(RecognitionUnavailable, match="未注册"):
        router.get_backend("material")


def test_missing_api_key_does_not_break_settings_or_startup_but_blocks_selected_backend(tmp_path):
    current = settings(tmp_path, recognition_enabled=True)
    assert RECOGNITION_REGISTRY.describe("qwen3-vl-plus", current).availability_status == "unavailable"
    router = RecognitionRouter(current)
    with pytest.raises(RecognitionUnavailable, match="API Key"):
        router.get_backend("classification")


def test_unimplemented_model_is_not_called_or_replaced(tmp_path):
    registry = RecognitionRegistry()
    registry.register(registration(availability_status="not_implemented", builder=None, reason="仅测试未实现状态"))
    with pytest.raises(RecognitionUnavailable, match="未实现"):
        RecognitionRouter(settings(tmp_path, recognition_model="test_model"), registry).get_backend("classification")


def test_initialization_failure_is_sanitized_cached_and_has_fresh_tracebacks(tmp_path):
    calls = []
    def broken(config):
        calls.append(True)
        raise RuntimeError("PRIVATE_API_KEY PRIVATE_IMAGE_BASE64")
    registry = RecognitionRegistry()
    registry.register(registration(builder=broken))
    router = RecognitionRouter(settings(tmp_path, recognition_model="test_model"), registry)
    failures = []
    for _ in range(3):
        with pytest.raises(RecognitionUnavailable) as caught:
            router.get_backend("classification")
        failures.append(caught.value)
    assert calls == [True] and len(set(map(id, failures))) == 3
    assert all("PRIVATE" not in "".join(traceback.format_exception(exc)) for exc in failures)


def test_instance_identity_and_capability_must_match_explicit_registration(tmp_path):
    registry = RecognitionRegistry()
    registry.register(registration(builder=lambda config: object()))
    with pytest.raises(RecognitionUnavailable, match="初始化失败"):
        registry.create("test_model", settings(tmp_path))


def test_model_configs_keep_credentials_independent_and_read_named_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SECOND_VLM_KEY", "TEST_SECOND_KEY")
    config = settings(tmp_path, recognition_api_key="TEST_DEFAULT_KEY", recognition_model_configs={
        "test_model": {"api_key_env": "SECOND_VLM_KEY", "api_base": "https://second.example/v1", "timeout": 17}})
    second = resolve_model_config(config, registration())
    assert second["api_key"] == "TEST_SECOND_KEY" and second["timeout"] == 17
    assert second["api_base"] == "https://second.example/v1" and second["model"] == "test-api-name"
    blank = resolve_model_config(settings(tmp_path, recognition_api_key="TEST_DEFAULT_KEY"), registration())
    assert blank["api_key"] == blank["api_base"] == ""
    assert "TEST_DEFAULT_KEY" not in repr(config) and "SECOND_VLM_KEY" not in repr(config)


def test_unverified_api_model_cannot_inherit_another_models_capabilities(tmp_path):
    current = settings(tmp_path, recognition_model_configs={"test_model": {"model": "unverified-name"}})
    with pytest.raises(RecognitionUnavailable, match="注册"):
        resolve_model_config(current, registration())


def test_signature_changes_with_relevant_configuration_but_contains_no_key(tmp_path):
    registry = RecognitionRegistry()
    registry.register(registration())
    current = settings(tmp_path, recognition_model="test_model", recognition_api_key="PRIVATE_API_KEY")
    signature = RecognitionRouter(current, registry).signature_for_task("classification")
    assert len(signature) == 64 and "PRIVATE" not in signature
    assert RecognitionRouter(replace(current, recognition_api_key="ANOTHER_KEY"), registry).signature_for_task("classification") == signature
    assert RecognitionRouter(replace(current, recognition_max_image_side=1000), registry).signature_for_task("classification") != signature


def test_environment_routes_and_profile_configs_parse_without_exposing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGITIZER_RECOGNITION_TASK_MODELS", '{"material":"test_model"}')
    monkeypatch.setenv("DIGITIZER_RECOGNITION_MODEL_CONFIGS", '{"test_model":{"api_key_env":"SECOND_KEY"}}')
    monkeypatch.setenv("DIGITIZER_RECOGNITION_API_KEY", "PRIVATE_KEY")
    current = Settings(data_dir=tmp_path)
    assert current.recognition_task_models == {"material": "test_model"}
    assert current.recognition_api_key == "PRIVATE_KEY" and "PRIVATE_KEY" not in repr(current)


@pytest.mark.parametrize("name,value", [("DIGITIZER_RECOGNITION_TASK_MODELS", "[]"),
                                       ("DIGITIZER_RECOGNITION_MODEL_CONFIGS", "bad JSON")])
def test_invalid_config_json_is_sanitized(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as caught:
        Settings(data_dir=tmp_path)
    assert value not in str(caught.value)


@pytest.mark.parametrize("overrides", [{"recognition_model_configs": {"test_model": {"api_key": "PRIVATE"}}},
                                       {"recognition_max_concurrency": 0}, {"recognition_timeout": float("nan")},
                                       {"recognition_task_models": {"material": "bad id"}}])
def test_unsafe_or_invalid_recognition_configuration_is_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        settings(tmp_path, **overrides)


def test_public_registry_metadata_does_not_expose_builders_or_keys():
    serialized = json.dumps(RECOGNITION_REGISTRY.get("qwen3-vl-plus").to_public_dict())
    assert "api_key" not in serialized and "builder" not in serialized and "configuration_check" not in serialized
