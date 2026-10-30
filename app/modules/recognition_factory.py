"""Explicit VLM registrations and task routing, independent of analysis tasks."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from threading import RLock
from typing import Literal

from app.config import Settings
from app.modules.recognition_backend import (
    RecognitionBackend, RecognitionCapabilities, RecognitionUnavailable, valid_identifier,
)


@dataclass(frozen=True)
class RecognitionRegistration:
    model_id: str
    provider: str
    model_name: str
    model_version: str
    capabilities: RecognitionCapabilities
    deployment_type: Literal["cloud", "local"]
    availability_status: Literal["available", "unavailable", "not_implemented"] = "available"
    description: str = ""
    builder: Callable[[Settings], RecognitionBackend] | None = field(default=None, repr=False, compare=False)
    configuration_check: Callable[[Settings], str | None] | None = field(default=None, repr=False, compare=False)
    reason: str = ""

    def __post_init__(self):
        if not valid_identifier(self.model_id):
            raise ValueError("视觉模型标识无效")
        for item in (self.provider, self.model_name, self.model_version):
            if not isinstance(item, str) or not item.strip() or len(item) > 128 or any(ord(c) < 32 for c in item):
                raise ValueError("视觉模型注册信息无效")
        if not isinstance(self.capabilities, RecognitionCapabilities):
            raise TypeError("视觉模型必须声明标准能力")
        if self.deployment_type not in {"cloud", "local"} or self.availability_status not in {"available", "unavailable", "not_implemented"}:
            raise ValueError("视觉模型部署方式或状态无效")
        if self.availability_status == "available" and not callable(self.builder):
            raise ValueError("已实现视觉模型必须提供延迟构造函数")
        if self.configuration_check is not None and not callable(self.configuration_check):
            raise TypeError("模型配置检查必须可调用")
        for value in (self.description, self.reason):
            if not isinstance(value, str) or len(value) > 2000 or any(ord(c) < 32 for c in value):
                raise ValueError("模型说明无效")

    def to_public_dict(self):
        return {"model_id": self.model_id, "provider": self.provider, "model_name": self.model_name,
                "model_version": self.model_version, "capabilities": asdict(self.capabilities),
                "deployment_type": self.deployment_type, "availability_status": self.availability_status,
                "description": self.description, "reason": self.reason}


class RecognitionRegistry:
    def __init__(self):
        self._models: dict[str, RecognitionRegistration] = {}
        self._lock = RLock()

    def register(self, registration: RecognitionRegistration):
        if not isinstance(registration, RecognitionRegistration):
            raise TypeError("需要标准视觉模型注册信息")
        with self._lock:
            if registration.model_id in self._models:
                raise ValueError("视觉模型ID已注册，不允许静默覆盖")
            self._models[registration.model_id] = registration

    def get(self, model_id: str) -> RecognitionRegistration:
        with self._lock:
            try:
                return self._models[model_id]
            except (KeyError, TypeError):
                raise RecognitionUnavailable("未注册的视觉模型，请检查模型与任务路由配置；不会回退到其他模型。") from None

    def describe(self, model_id: str, settings: Settings | None = None) -> RecognitionRegistration:
        registration = self.get(model_id)
        if settings is not None and registration.availability_status == "available" and registration.configuration_check:
            try:
                reason = registration.configuration_check(settings)
            except Exception:
                raise RecognitionUnavailable("视觉模型配置检查失败，请检查服务端配置；不会回退到其他模型。") from None
            if reason:
                return replace(registration, availability_status="unavailable", reason=reason)
        return registration

    def list_models(self, settings: Settings | None = None):
        with self._lock:
            ids = tuple(self._models)
        return tuple(self.describe(model_id, settings) for model_id in ids)

    def create(self, model_id: str, settings: Settings) -> RecognitionBackend:
        registration = self.describe(model_id, settings)
        if registration.availability_status != "available":
            raise RecognitionUnavailable(registration.reason or "所选视觉模型尚不可用或未实现；不会回退。")
        try:
            backend = registration.builder(settings)
            if (not isinstance(backend, RecognitionBackend) or backend.model_id != registration.model_id
                    or backend.model_version != registration.model_version or backend.capabilities != registration.capabilities):
                raise TypeError("Backend identity/capabilities differ from registration")
            return backend
        except RecognitionUnavailable:
            raise
        except Exception:
            raise RecognitionUnavailable("所选视觉模型初始化失败，请检查依赖和配置；不会回退到其他模型。") from None


def resolve_model_config(settings: Settings, registration: RecognitionRegistration) -> dict:
    """Private configuration only; keys are read from the specified environment."""
    override = settings.recognition_model_configs.get(registration.model_id, {})
    is_default = registration.model_id == settings.recognition_model
    api_key_env = override.get("api_key_env", "DIGITIZER_RECOGNITION_API_KEY" if is_default else "")
    api_key = (settings.recognition_api_key if is_default and "api_key_env" not in override
               else os.environ.get(api_key_env, "").strip())
    model = override.get("model", registration.model_name)
    if model != registration.model_name:
        raise RecognitionUnavailable("API模型名称与已核实能力的注册信息不一致，请为该模型单独注册，不能继承其他模型能力。")
    return {"api_base": override.get("api_base", settings.recognition_api_base if is_default else ""),
            "api_key": api_key, "model": model, "timeout": override.get("timeout", settings.recognition_timeout)}


QWEN_CAPABILITIES = RecognitionCapabilities(image_input=True, multi_image_input=True, structured_output=True)


def _qwen_configuration(settings: Settings) -> str | None:
    if settings.recognition_backend != "openai_compatible":
        return "当前视觉调用适配器未注册，请检查DIGITIZER_RECOGNITION_BACKEND。"
    config = resolve_model_config(settings, RECOGNITION_REGISTRY.get("qwen3-vl-plus"))
    if not config["api_key"]:
        return "尚未配置视觉模型API Key，请设置服务端环境变量后重启；现有图片功能仍可使用。"
    if not config["api_base"]:
        return "尚未配置视觉模型API Base URL。"
    return None


def _build_qwen(settings: Settings) -> RecognitionBackend:
    from app.modules.openai_compatible_backend import OpenAICompatibleRecognitionBackend

    registration = RECOGNITION_REGISTRY.get("qwen3-vl-plus")
    config = resolve_model_config(settings, registration)
    return OpenAICompatibleRecognitionBackend(
        model_id=registration.model_id, model_version=registration.model_version,
        model_name=config["model"], api_base=config["api_base"], api_key=config["api_key"],
        capabilities=registration.capabilities, timeout=config["timeout"],
        max_images=settings.recognition_max_images, max_image_side=settings.recognition_max_image_side,
        max_concurrency=settings.recognition_max_concurrency, min_interval=settings.recognition_min_interval,
        max_output_tokens=settings.recognition_max_output_tokens, disable_thinking=True,
    )


RECOGNITION_REGISTRY = RecognitionRegistry()
RECOGNITION_REGISTRY.register(RecognitionRegistration(
    model_id="qwen3-vl-plus", provider="DashScope", model_name="qwen3-vl-plus", model_version="service-alias",
    capabilities=QWEN_CAPABILITIES, deployment_type="cloud", builder=_build_qwen,
    configuration_check=_qwen_configuration,
    description="Qwen3-VL-Plus非思考图文JSON Object适配；不声明服务端JSON Schema约束、视频或多轮接口。",
))


class RecognitionRouter:
    """One instance per processor; selected adapters are reused across requests."""

    def __init__(self, settings: Settings, registry: RecognitionRegistry | None = None):
        self.settings = settings
        self.registry = registry if registry is not None else RECOGNITION_REGISTRY
        self._backends: dict[str, RecognitionBackend] = {}
        self._errors: dict[str, str] = {}
        self._lock = RLock()

    def model_id_for_task(self, task_id: str) -> str:
        if not valid_identifier(task_id):
            raise RecognitionUnavailable("视觉任务标识无效。")
        return self.settings.recognition_task_models.get(task_id, self.settings.recognition_model)

    def get_backend(self, task_id: str) -> RecognitionBackend:
        selected = self.model_id_for_task(task_id)
        with self._lock:
            if selected in self._errors:
                raise RecognitionUnavailable(self._errors[selected]) from None
            if selected not in self._backends:
                try:
                    self._backends[selected] = self.registry.create(selected, self.settings)
                except RecognitionUnavailable as exc:
                    self._errors[selected] = str(exc)
                    raise RecognitionUnavailable(str(exc)) from None
            return self._backends[selected]

    def signature_for_task(self, task_id: str) -> str:
        selected = self.model_id_for_task(task_id)
        registration = self.registry.get(selected)
        override = self.settings.recognition_model_configs.get(selected, {})
        identity = {
            "registration": registration.to_public_dict(), "backend": self.settings.recognition_backend,
            "api_base": override.get("api_base", self.settings.recognition_api_base if selected == self.settings.recognition_model else ""),
            "api_model": override.get("model", registration.model_name),
            "max_image_side": self.settings.recognition_max_image_side,
            "max_output_tokens": self.settings.recognition_max_output_tokens,
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
