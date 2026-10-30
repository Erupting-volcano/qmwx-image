"""A small task definition: prompt + typed input/output + capability requirements."""

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel, ValidationError

from app.models.object_understanding import (
    FusionInput, ModelCallMetadata, SingleImageTaskInput, validate_sources,
    validate_fusion_against_observations,
)
from app.modules.recognition_backend import (
    RecognitionBackend, RecognitionCapabilities, RecognitionImage, RecognitionRequest,
    RecognitionResult, check_recognition_capabilities,
)
from app.prompts.object_understanding import compose_prompts, get_prompt


@dataclass(frozen=True)
class TaskExecution:
    data: BaseModel
    model_metadata: ModelCallMetadata
    prompt_versions: dict[str, str]


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    description: str
    output_model: Type[BaseModel]
    input_model: Type[BaseModel] = SingleImageTaskInput

    @property
    def prompt_version(self) -> str:
        return get_prompt(self.task_id).version

    @property
    def required_capabilities(self) -> RecognitionCapabilities:
        return RecognitionCapabilities(image_input=self.input_model is SingleImageTaskInput, structured_output=True)

    def schema(self) -> dict:
        return self.output_model.model_json_schema()

    def validate_output(self, data: dict, allowed_asset_ids) -> BaseModel:
        try:
            validated = self.output_model.model_validate(data.model_dump() if isinstance(data, BaseModel) else data)
        except (ValidationError, TypeError):
            # ValidationError can embed untrusted provider text. Do not put a raw
            # response (or echoed photo Base64) in downstream exception logs.
            raise ValueError("任务输出未通过结构校验，请根据任务与错误编号检查；原始模型输出不会写入日志。") from None
        validate_sources(validated, allowed_asset_ids)
        return validated

    def execute(self, backend: RecognitionBackend, input_data, images: tuple[RecognitionImage, ...] = ()) -> TaskExecution:
        """Explicit standalone execution, useful for a separately routed task.

        Normal uploads compose tasks in the processor to avoid five image calls.
        This method never chooses or instantiates a provider.
        """
        input_data = self.input_model.model_validate(input_data)
        if isinstance(input_data, SingleImageTaskInput):
            if (not images or not any(item.role == "original" for item in images)
                    or any(item.asset_id != input_data.asset_id for item in images)):
                raise ValueError("单图任务必须提供所属原图及可选的同图辅助裁剪")
            allowed = {input_data.asset_id}
        else:
            if images:
                raise ValueError("本版语义融合使用已验证的观察，不重复发送图片")
            allowed = {item.asset_id for item in input_data.image_observations}
        composed = compose_prompts((self.task_id,), model_id=backend.model_id)
        request = RecognitionRequest(
            task_id=self.task_id, instruction=composed.instruction, images=tuple(images),
            output_schema=self.schema(), context=input_data.model_dump(mode="json"),
            required_capabilities=self.required_capabilities,
        )
        check_recognition_capabilities(request, backend.capabilities)
        result = backend.analyze(request)
        if (not isinstance(result, RecognitionResult) or result.model_id != backend.model_id
                or result.model_version != backend.model_version):
            raise ValueError("模型必须返回对应身份的统一结果")
        output = self.validate_output(result.data, allowed)
        if isinstance(input_data, FusionInput):
            validate_fusion_against_observations(output, input_data.image_observations)
        return TaskExecution(output, ModelCallMetadata(
            model_id=result.model_id, model_version=result.model_version, task_ids=[self.task_id],
            processing_time=result.processing_time, warnings=list(result.warnings), metadata=result.metadata,
        ), composed.versions)
