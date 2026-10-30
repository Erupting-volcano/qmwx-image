"""Explicit model registrations and lazy construction; never choose a fallback.

Metadata queries do not import backend modules, load models, contact providers,
or inspect credentials outside the selected configuration. A declared capability
for a not_implemented registration describes the future adapter, not readiness.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
import re
from threading import RLock
from typing import Literal

from app.config import Settings
from app.modules.segmentation_backend import (
    BackendInitializationError, NotImplementedBackendError, SegmentationBackend,
    SegmentationCapabilities, SegmentationUnavailable, UnknownBackendError,
)


BackendStatus = Literal["available", "unavailable", "not_implemented"]
BackendBuilder = Callable[[Settings], SegmentationBackend]
ConfigurationCheck = Callable[[Settings], str | None]


@dataclass(frozen=True)
class BackendRegistration:
    model_id: str
    name: str
    model_name: str
    model_version: str
    description: str
    capabilities: SegmentationCapabilities
    deployment: Literal["cloud", "local"]
    status: BackendStatus
    reason: str = ""
    builder: BackendBuilder | None = field(default=None, repr=False, compare=False)
    configuration_check: ConfigurationCheck | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for value in (self.model_id, self.model_name):
            if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value):
                raise ValueError("模型注册标识无效")
        for value in (self.name, self.model_version, self.description):
            if not isinstance(value, str) or not value.strip() or len(value) > 2048 or any(ord(c) < 32 for c in value):
                raise ValueError("模型注册信息需要有效的短文本")
        if len(self.model_version) > 128:
            raise ValueError("模型版本标识不得超过128个字符")
        if not isinstance(self.reason, str) or len(self.reason) > 2048 or any(ord(c) < 32 for c in self.reason):
            raise ValueError("模型状态说明无效")
        if not isinstance(self.capabilities, SegmentationCapabilities):
            raise TypeError("注册模型必须显式声明能力")
        if self.deployment not in {"local", "cloud"} or self.status not in {"available", "unavailable", "not_implemented"}:
            raise ValueError("模型部署方式或状态无效")
        if self.status == "available" and not callable(self.builder):
            raise ValueError("可用模型必须提供构造函数")
        if self.builder is not None and not callable(self.builder):
            raise TypeError("模型构造函数必须可调用")
        if self.configuration_check is not None and not callable(self.configuration_check):
            raise TypeError("模型配置检查必须可调用")

    def to_public_dict(self) -> dict:
        """Only safe registration metadata; no credentials, paths, or callbacks."""
        return {
            "model_id": self.model_id, "name": self.name, "model_name": self.model_name,
            "model_version": self.model_version, "description": self.description,
            "capabilities": asdict(self.capabilities), "deployment": self.deployment,
            "status": self.status, "reason": self.reason,
        }


class SegmentationRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, BackendRegistration] = {}
        self._lock = RLock()

    def register(self, registration: BackendRegistration) -> None:
        if not isinstance(registration, BackendRegistration):
            raise TypeError("需要标准模型注册信息")
        with self._lock:
            if registration.model_id in self._registrations:
                raise ValueError("分割模型标识已经注册，不允许静默覆盖")
            self._registrations[registration.model_id] = registration

    def get(self, model_id: str) -> BackendRegistration:
        with self._lock:
            try:
                return self._registrations[model_id]
            except (KeyError, TypeError):
                raise UnknownBackendError("未注册的分割模型，请检查DIGITIZER_SEGMENTATION_BACKEND；不会切换到其他模型。") from None

    def describe(self, model_id: str, settings: Settings | None = None) -> BackendRegistration:
        registration = self.get(model_id)
        if settings is not None and registration.status == "available" and registration.configuration_check:
            try:
                reason = registration.configuration_check(settings)
            except Exception:
                raise BackendInitializationError("所选分割模型配置检查失败，请检查服务端配置；不会切换到其他模型。") from None
            if reason:
                # Built-in/custom checks must return safe messages, never values.
                return replace(registration, status="unavailable", reason=reason)
        return registration

    def list_models(self, settings: Settings | None = None) -> tuple[BackendRegistration, ...]:
        with self._lock:
            ids = tuple(self._registrations)
        return tuple(self.describe(model_id, settings) for model_id in ids)

    def create(self, model_id: str, settings: Settings) -> SegmentationBackend:
        registration = self.describe(model_id, settings)
        if registration.status == "not_implemented":
            raise NotImplementedBackendError(registration.reason or "所选分割模型尚未实现，不会产生掩码或切换到其他模型。")
        if registration.status != "available":
            raise SegmentationUnavailable(registration.reason or "所选分割模型尚不可用，不会切换到其他模型。")
        try:
            backend = registration.builder(settings)
            if (not isinstance(backend, SegmentationBackend)
                    or backend.model_id != registration.model_id
                    or backend.model_name != registration.model_name
                    or backend.model_version != registration.model_version
                    or backend.capabilities != registration.capabilities):
                raise TypeError("Backend identity or capabilities differ from registration")
            return backend
        except SegmentationUnavailable:
            raise
        except Exception:
            raise BackendInitializationError("所选分割模型初始化失败，请检查服务端配置与依赖；不会切换到其他模型。") from None


def _baidu_configuration(settings: Settings) -> str | None:
    if not settings.segmentation_api_key.strip() or not settings.segmentation_secret_key.strip():
        return "尚未配置百度API Key和Secret Key，请配置后重启服务；不会切换到其他模型或伪造掩码。"
    return None


def _build_baidu(settings: Settings) -> SegmentationBackend:
    from app.modules.baidu_backend import BaiduSegmentationBackend

    return BaiduSegmentationBackend(settings.segmentation_api_key, settings.segmentation_secret_key,
                                    settings.segmentation_timeout_seconds)


BAIDU_CAPABILITIES = SegmentationCapabilities(box_prompt=True, automatic_mask=True)
SAM2_CAPABILITIES = SegmentationCapabilities(
    box_prompt=True, point_prompt=True, mask_prompt=True, automatic_mask=True,
    multiple_masks=True, quality_score=True, iterative_refinement=True,
)
SAM2_NOT_IMPLEMENTED = "SAM 2.1 Hiera Tiny仅登记了能力和配置接口，尚未实现或部署；本次不会加载依赖、下载权重、生成掩码或改用百度。"


SEGMENTATION_REGISTRY = SegmentationRegistry()
SEGMENTATION_REGISTRY.register(BackendRegistration(
    model_id="baidu", name="百度智能云智能抠图", model_name="baidu_intelligent_segmentation",
    model_version="service-unversioned",
    description="百度云矩形框提示与显式自动主体提取；服务未公开模型架构版本，不提供可靠质量分数。",
    capabilities=BAIDU_CAPABILITIES, deployment="cloud", status="available",
    reason="适配器已实现；实际调用仍取决于应用授权、额度及网络。",
    builder=_build_baidu, configuration_check=_baidu_configuration,
))
SEGMENTATION_REGISTRY.register(BackendRegistration(
    model_id="sam2.1_hiera_tiny", name="Meta SAM 2.1 Hiera Tiny", model_name="sam2.1_hiera_tiny",
    model_version="2.1", description="未来本地模型接入位置；所列能力是预留定义，不代表已经实现推理。",
    capabilities=SAM2_CAPABILITIES, deployment="local", status="not_implemented", reason=SAM2_NOT_IMPLEMENTED,
))


def get_backend_registration(model_id: str) -> BackendRegistration:
    return SEGMENTATION_REGISTRY.get(model_id)


def describe_backend(model_id: str, settings: Settings | None = None) -> BackendRegistration:
    return SEGMENTATION_REGISTRY.describe(model_id, settings)


def create_backend(settings: Settings) -> SegmentationBackend:
    return SEGMENTATION_REGISTRY.create(settings.segmentation_backend, settings)
