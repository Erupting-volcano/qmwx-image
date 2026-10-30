"""Mock HTTP transport tests, never evidence of real model recognition quality."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import base64
import json
from threading import Lock
import time
import traceback
from urllib.error import HTTPError, URLError
from uuid import uuid4

from PIL import Image
import pytest

from app.modules import openai_compatible_backend as adapter
from app.modules.openai_compatible_backend import OpenAICompatibleRecognitionBackend
from app.modules.recognition_backend import (
    RecognitionCapabilities, RecognitionError, RecognitionImage, RecognitionRequest, RecognitionResult,
    RecognitionUnavailable, UnsupportedRecognitionCapability,
)


CAPS = RecognitionCapabilities(image_input=True, multi_image_input=True, structured_output=True)
SECRET = "TEST_PRIVATE_VLM_KEY"


class Response(BytesIO):
    status = 200

    def __init__(self, value):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())


class Opener:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("No live request permitted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return Response(response)


def response(data=None, *, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(data or {"category": "test"})}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14, "private": SECRET}}


def backend(*responses, **overrides):
    opener = Opener(*responses)
    values = dict(model_id="qwen3-vl-plus", model_version="service-alias", model_name="qwen3-vl-plus",
                  api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=SECRET,
                  capabilities=CAPS, min_interval=0, disable_thinking=True, opener=opener)
    values.update(overrides)
    return OpenAICompatibleRecognitionBackend(**values), opener


@pytest.fixture
def request_image():
    with Image.new("RGB", (96, 64), (50, 80, 100)) as image:
        yield RecognitionRequest("classification", "只观察图片，输出JSON，不要猜测。",
                                 (RecognitionImage(uuid4(), image),), {"type": "object"}, {"name": "测试"})


def test_standard_contract_uses_image_data_and_json_without_mutating_input(request_image):
    model, opener = backend(response())
    before = request_image.images[0].image.tobytes()
    assert opener.calls == []
    result = model.analyze(request_image)
    assert result.data == {"category": "test"}
    assert result.model_id == "qwen3-vl-plus" and result.model_version == "service-alias"
    assert result.processing_time >= 0
    assert result.metadata == {"qwen3-vl-plus": {"usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
                                               "output_mode": "json_object"}}
    assert SECRET not in repr(result) and SECRET not in json.dumps(result.metadata)
    assert len(opener.calls) == 1
    sent, timeout = opener.calls[0]
    assert sent.full_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert timeout == 60 and sent.get_header("Authorization") == "Bearer " + SECRET
    assert SECRET not in sent.full_url
    body = json.loads(sent.data)
    assert body["response_format"] == {"type": "json_object"}
    assert body["enable_thinking"] is False and body["max_tokens"] == 8192 and body["stream"] is False
    parts = body["messages"][1]["content"]
    assert str(request_image.images[0].asset_id) in parts[1]["text"]
    with Image.open(BytesIO(base64.b64decode(parts[2]["image_url"]["url"].split(",")[1]))) as encoded:
        assert encoded.mode == "RGB" and encoded.size == (96, 64) and encoded.format == "JPEG"
        assert not encoded.getexif()
    assert request_image.images[0].image.tobytes() == before


def test_text_only_fusion_request_is_supported_without_discarding_context():
    model, opener = backend(response({"consensus": []}))
    result = model.analyze(RecognitionRequest("multi_view_fusion", "融合观察并输出JSON", context={"images": [{"view": "unknown"}]}))
    assert result.data == {"consensus": []}
    parts = json.loads(opener.calls[0][0].data)["messages"][1]["content"]
    assert len(parts) == 1 and "unknown" in parts[0]["text"]


def test_missing_key_never_sends_a_request(request_image):
    model, opener = backend(api_key="")
    with pytest.raises(RecognitionUnavailable, match="API Key"):
        model.analyze(request_image)
    assert not opener.calls


@pytest.mark.parametrize("capability", ["json_schema", "video_input", "multi_turn", "high_resolution_image"])
def test_unimplemented_capability_is_rejected_before_network_or_credentials(request_image, capability):
    model, opener = backend(api_key="")
    required = RecognitionCapabilities(**{capability: True})
    with pytest.raises(UnsupportedRecognitionCapability):
        model.analyze(replace(request_image, required_capabilities=required))
    assert not opener.calls


def test_multiple_images_are_not_silently_dropped(request_image):
    model, opener = backend(capabilities=RecognitionCapabilities(image_input=True, structured_output=True))
    with pytest.raises(UnsupportedRecognitionCapability, match="多图"):
        model.analyze(replace(request_image, images=request_image.images * 2))
    assert not opener.calls


@pytest.mark.parametrize("capability", ["json_schema", "high_resolution_image", "multi_turn", "video_input"])
def test_adapter_cannot_claim_unimplemented_interfaces_by_flipping_flags(capability):
    capabilities = RecognitionCapabilities(image_input=True, structured_output=True, **{capability: True})
    with pytest.raises(RecognitionUnavailable, match="独立适配"):
        backend(capabilities=capabilities)


def test_auxiliary_crop_is_explicit_and_original_is_retained(request_image):
    model, opener = backend(response())
    with request_image.images[0].image.crop((10, 10, 40, 40)) as crop:
        auxiliary = RecognitionImage(request_image.images[0].asset_id, crop, "target_region")
        model.analyze(replace(request_image, images=(*request_image.images, auxiliary)))
    parts = json.loads(opener.calls[0][0].data)["messages"][1]["content"]
    assert len([part for part in parts if part["type"] == "image_url"]) == 2
    assert "original" in parts[1]["text"] and "target_region" in parts[3]["text"]


def test_send_copy_is_scaled_without_altering_original(request_image):
    model, opener = backend(response(), max_image_side=48)
    result = model.analyze(request_image)
    assert result.warnings and request_image.images[0].image.size == (96, 64)
    part = json.loads(opener.calls[0][0].data)["messages"][1]["content"][2]
    with Image.open(BytesIO(base64.b64decode(part["image_url"]["url"].split(",")[1]))) as image:
        assert image.size == (48, 32)


@pytest.mark.parametrize("overrides", [{"max_images": 1}, {"max_image_pixels": 20}])
def test_image_count_and_pixels_are_bounded_before_network(request_image, overrides):
    model, opener = backend(**overrides)
    request = replace(request_image, images=request_image.images * 2) if "max_images" in overrides else request_image
    with pytest.raises(RecognitionError):
        model.analyze(request)
    assert not opener.calls


@pytest.mark.parametrize("status", [301, 307, 400, 401, 403, 429, 500, 503])
def test_http_failures_are_safe_and_never_retried(request_image, status):
    error = HTTPError("https://secret.example/" + SECRET, status, SECRET, {}, BytesIO(SECRET.encode()))
    model, opener = backend(error)
    with pytest.raises(RecognitionError) as caught:
        model.analyze(request_image)
    rendered = "".join(traceback.format_exception(caught.value))
    assert SECRET not in str(caught.value) and "secret.example" not in rendered
    assert len(opener.calls) == 1


@pytest.mark.parametrize("error", [URLError(SECRET), TimeoutError(SECRET), OSError(SECRET), RuntimeError(SECRET)])
def test_transport_errors_do_not_expose_payloads_or_credentials(request_image, error):
    model, opener = backend(error)
    with pytest.raises(RecognitionError) as caught:
        model.analyze(request_image)
    assert SECRET not in "".join(traceback.format_exception(caught.value)) and len(opener.calls) == 1


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", None])
def test_incomplete_outputs_are_not_adopted_or_repaired_with_paid_calls(request_image, finish):
    model, opener = backend(response(finish=finish))
    with pytest.raises(RecognitionError, match="截断"):
        model.analyze(request_image)
    assert len(opener.calls) == 1


@pytest.mark.parametrize("content", ["not JSON", "```json\n{}\n```", "[]", '{"a":1,"a":2}', '{"a":NaN}'])
def test_invalid_model_json_is_rejected_without_invented_output(request_image, content):
    payload = response()
    payload["choices"][0]["message"]["content"] = content
    model, opener = backend(payload)
    with pytest.raises(RecognitionError):
        model.analyze(request_image)
    assert len(opener.calls) == 1


@pytest.mark.parametrize("payload", [{"error": {"message": SECRET}}, {"choices": []}, {"choices": [None]},
                                     b"invalid JSON", b"[]", b'{"choices":NaN}'])
def test_invalid_provider_response_is_safe(request_image, payload):
    model, opener = backend(payload)
    with pytest.raises(RecognitionError) as caught:
        model.analyze(request_image)
    assert SECRET not in str(caught.value) and len(opener.calls) == 1


def test_oversized_http_response_is_rejected(request_image, monkeypatch):
    monkeypatch.setattr(adapter, "MAX_RESPONSE_BYTES", 20)
    model, opener = backend(response())
    with pytest.raises(RecognitionError, match="超过"):
        model.analyze(request_image)


@pytest.mark.parametrize("base", ["http://example.com/v1", "https://user:pass@example.com/v1", "https://example.com/v1?key=secret",
                                 "https://example.com/v1#fragment", "https://{WorkspaceId}.example.com/v1", "not-url"])
def test_base_url_must_be_explicit_trusted_https_without_embedded_credentials(base):
    with pytest.raises(RecognitionUnavailable):
        backend(api_base=base)


def test_semaphore_is_released_after_a_failed_call(request_image):
    model, opener = backend(TimeoutError(SECRET), response(), max_concurrency=1)
    with pytest.raises(RecognitionError):
        model.analyze(request_image)
    assert model.analyze(request_image).data == {"category": "test"}


def test_configured_rate_limit_is_applied_between_calls(request_image, monkeypatch):
    now, sleeps = [0.0], []
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now[0])
    def sleep(value):
        sleeps.append(value)
        now[0] += value
    monkeypatch.setattr(adapter.time, "sleep", sleep)
    model, _ = backend(response(), response(), min_interval=0.5)
    model.analyze(request_image)
    model.analyze(request_image)
    assert sleeps == [0.5]


def test_concurrent_calls_are_bounded(request_image):
    active, peaks = [0], []
    lock = Lock()
    class ConcurrentOpener:
        def open(self, request, timeout):
            with lock:
                active[0] += 1
                peaks.append(active[0])
            time.sleep(0.01)
            with lock:
                active[0] -= 1
            return Response(response())
    model, _ = backend(opener=ConcurrentOpener(), max_concurrency=2)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: model.analyze(request_image), range(6)))
    assert len(results) == 6 and max(peaks) <= 2


def test_contracts_validate_bounded_metadata_and_do_not_repr_private_data():
    with Image.new("RGB", (4, 4)) as image:
        request = RecognitionRequest("classification", SECRET, (RecognitionImage(uuid4(), image),), context={"private": SECRET})
        assert SECRET not in repr(request)
    result = RecognitionResult({"private": SECRET}, "test_model", "test", 0.1)
    assert SECRET not in repr(result)
    for metadata in ({"bad space": {}}, {"model": {"raw": b"binary"}}, {"model": {"large": "x" * 20000}}):
        with pytest.raises(ValueError):
            RecognitionResult({}, "test_model", "test", 0, metadata=metadata)


@pytest.mark.parametrize("kwargs", [{"image_input": 1}, {"structured_output": "true"}])
def test_capabilities_cannot_be_inferred_from_truthy_values(kwargs):
    with pytest.raises(ValueError):
        RecognitionCapabilities(**kwargs)
