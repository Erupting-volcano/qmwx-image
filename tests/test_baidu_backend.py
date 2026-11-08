"""Official Baidu HTTP contract and transforms, without network or paid calls."""

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import random
from threading import Event
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageDraw
import pytest

from app.modules import baidu_backend as api
from app.modules.baidu_backend import BaiduSegmentationBackend
from app.modules.segmentation_backend import (
    AutomaticPrompt, BoxPrompt, MaskPrompt, PointPrompt, SegmentationAPIError,
    SegmentationBackend, SegmentationRequest, SegmentationUnavailable,
    TextPrompt, UnsupportedPromptError,
)


TOKEN = "TEST_ACCESS_TOKEN_PRIVATE"
KEY = "TEST_API_KEY_PRIVATE"
SECRET = "TEST_SECRET_KEY_PRIVATE"


class Response(BytesIO):
    def __init__(self, body, *, status=200):
        super().__init__(json.dumps(body).encode() if not isinstance(body, bytes) else body)
        self.status = status


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, *, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("Unexpected API request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, Response) else Response(response)


def token(value=TOKEN, expires=3600):
    return {"access_token": value, "expires_in": expires}


def png64(image):
    with BytesIO() as stream:
        image.save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode()


def mask_response(size=(256, 192), *, value=0, gray=False):
    with Image.new("L", size, value) as mask:
        if not value:
            ImageDraw.Draw(mask).rectangle((20, 30, min(size[0] - 1, 200), min(size[1] - 1, 150)), fill=128 if gray else 255)
        return {"image": png64(mask)}


def backend(*responses, **kwargs):
    opener = Opener(*responses)
    return BaiduSegmentationBackend(KEY, SECRET, opener=opener, **kwargs), opener


@pytest.fixture
def rgb():
    with Image.new("RGB", (256, 192), (90, 120, 150)) as image:
        yield image


BOX = (16, 20, 220, 170)


def test_initialization_is_lazy_and_sends_official_control_contract(rgb):
    instance, opener = backend(token(), mask_response(), timeout_seconds=17)
    assert opener.calls == []
    original = rgb.tobytes()
    result = instance.predict(rgb, BOX)
    with result.mask:
        assert result.mask.size == rgb.size
        assert set(result.mask.tobytes()) == {0, 255}
        assert result.score is None
        assert result.warnings == ()
    assert rgb.tobytes() == original
    assert instance.model_name == "baidu_intelligent_segmentation"
    assert len(opener.calls) == 2
    auth, timeout = opener.calls[0]
    assert timeout == 17
    assert auth.full_url == api.TOKEN_URL
    assert auth.get_method() == "POST"
    assert auth.get_header("Content-type") == "application/x-www-form-urlencoded"
    assert parse_qs(auth.data.decode()) == {
        "grant_type": ["client_credentials"], "client_id": [KEY], "client_secret": [SECRET],
    }
    request, _ = opener.calls[1]
    assert request.full_url.startswith(api.SEGMENT_URL + "?")
    assert parse_qs(urlsplit(request.full_url).query) == {"access_token": [TOKEN]}
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data)
    assert set(payload) == {"image", "method", "position", "return_form", "refine_mask"}
    assert payload["method"] == "control"
    assert payload["position"] == [[[16, 20], [220, 170]]]
    assert payload["return_form"] == "mask"
    assert payload["refine_mask"] == "false"
    with Image.open(BytesIO(base64.b64decode(payload["image"]))) as sent:
        assert sent.size == rgb.size
        assert sent.format == "PNG"
        assert sent.tobytes() == original


def test_token_is_cached_across_predictions(rgb):
    instance, opener = backend(token(), mask_response(), mask_response())
    for _ in range(2):
        instance.predict(rgb, BOX).mask.close()
    assert len(opener.calls) == 3
    assert sum(request.full_url == api.TOKEN_URL for request, _ in opener.calls) == 1


def test_token_refreshes_before_expiry(monkeypatch, rgb):
    now = [1000.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: now[0])
    instance, opener = backend(token(expires=120), mask_response(), token("refreshed"), mask_response())
    instance.predict(rgb, BOX).mask.close()
    now[0] += 61
    instance.predict(rgb, BOX).mask.close()
    assert len(opener.calls) == 4
    assert parse_qs(urlsplit(opener.calls[3][0].full_url).query)["access_token"] == ["refreshed"]


def test_concurrent_first_use_fetches_only_one_token():
    entered, release = Event(), Event()

    class BlockingOpener(Opener):
        def open(self, request, *, timeout):
            entered.set()
            assert release.wait(3), "Test failed to release simulated token response"
            return super().open(request, timeout=timeout)

    opener = BlockingOpener(token())
    instance = BaiduSegmentationBackend(KEY, SECRET, opener=opener)
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(instance._access_token)
        assert entered.wait(3), "No simulated token request received"
        others = [executor.submit(instance._access_token) for _ in range(3)]
        release.set()
        assert [future.result(timeout=3) for future in [first, *others]] == [TOKEN] * 4
    assert len(opener.calls) == 1


def test_large_image_box_and_mask_map_to_original():
    instance, opener = backend(token(), mask_response((3000, 1500)))
    with Image.new("RGB", (4000, 2000), "white") as image:
        result = instance.predict(image, (400, 200, 3600, 1800))
        with result.mask:
            assert result.mask.size == (4000, 2000)
            assert result.mask.getpixel((30, 45)) == 255
            assert result.mask.getpixel((0, 0)) == 0
        assert any("等比缩小" in warning for warning in result.warnings)
    payload = json.loads(opener.calls[1][0].data)
    assert payload["position"] == [[[300, 150], [2700, 1350]]]
    with Image.open(BytesIO(base64.b64decode(payload["image"]))) as sent:
        assert sent.size == (3000, 1500)


def test_small_image_padding_is_removed_without_altering_original():
    with Image.new("L", (128, 128)) as mask:
        ImageDraw.Draw(mask).rectangle((48, 52, 79, 75), fill=255)
        instance, opener = backend(token(), {"image": png64(mask)})
    with Image.new("RGB", (32, 24), (7, 19, 31)) as image:
        original = image.tobytes()
        result = instance.predict(image, (0, 0, 32, 24))
        with result.mask:
            assert result.mask.size == (32, 24)
            assert result.mask.getextrema() == (255, 255)
        assert image.tobytes() == original
    assert json.loads(opener.calls[1][0].data)["position"] == [[[48, 52], [80, 76]]]
    assert any("白色边距" in item for item in result.warnings)


def test_box_on_image_edges_is_adjusted_and_disclosed(rgb):
    instance, opener = backend(token(), mask_response())
    result = instance.predict(rgb, (0, 0, 256, 192))
    result.mask.close()
    assert json.loads(opener.calls[1][0].data)["position"] == [[[1, 1], [255, 191]]]
    assert any("向内调整1像素" in item for item in result.warnings)


@pytest.mark.parametrize("box", [
    (1, 1, 5, 5), (-1, 0, 20, 20), (20, 1, 10, 20), (0, 0, 257, 20),
    (0, 0, float("nan"), 20), (0, 0, float("inf"), 20), (False, 0, 20, 20), (0, 1, 2),
])
def test_invalid_or_too_small_box_never_spends_api_call(rgb, box):
    instance, opener = backend()
    with pytest.raises(SegmentationAPIError):
        instance.predict(rgb, box)
    assert opener.calls == []


def test_extreme_aspect_ratio_does_not_expand_too_small_box():
    instance, opener = backend()
    with Image.new("RGB", (4000, 8)) as image, pytest.raises(SegmentationAPIError, match="小于10"):
        instance.predict(image, (0, 0, 4000, 8))
    assert opener.calls == []


def test_encoded_image_size_limit_precedes_auth(monkeypatch, rgb):
    monkeypatch.setattr(api, "MAX_IMAGE_BASE64_BYTES", 4)
    instance, opener = backend()
    with pytest.raises(SegmentationAPIError, match="安全大小"):
        instance.predict(rgb, BOX)
    assert opener.calls == []


def test_compression_fallback_is_disclosed_and_keeps_original(monkeypatch):
    rng = random.Random(41)
    with Image.frombytes("RGB", (256, 192), rng.randbytes(256 * 192 * 3)) as image:
        original = image.tobytes()
        monkeypatch.setattr(api, "MAX_IMAGE_BASE64_BYTES", 100_000)
        instance, opener = backend(token(), mask_response())
        result = instance.predict(image, BOX)
        result.mask.close()
        assert any("JPEG压缩副本" in item for item in result.warnings)
        with Image.open(BytesIO(base64.b64decode(json.loads(opener.calls[1][0].data)["image"]))) as sent:
            assert sent.format == "JPEG"
        assert image.tobytes() == original


@pytest.mark.parametrize("response", [
    {}, {"image": "not base64!"}, {"image": base64.b64encode(b"broken PNG").decode()},
    {"image_url": "https://attacker.invalid/private"},
])
def test_missing_or_corrupt_mask_is_safely_rejected(rgb, response):
    instance, opener = backend(token(), response)
    with pytest.raises(SegmentationAPIError):
        instance.predict(rgb, BOX)
    assert len(opener.calls) == 2  # Never fetch an arbitrary response URL.


@pytest.mark.parametrize("mode,size", [("L", (255, 192)), ("RGB", (256, 192)), ("RGBA", (256, 192))])
def test_mask_mode_and_dimensions_are_checked_before_decode(rgb, mode, size):
    with Image.new(mode, size) as mask:
        instance, _ = backend(token(), {"image": png64(mask)})
    with pytest.raises(SegmentationAPIError, match="尺寸.*不一致|编码模式暂不支持"):
        instance.predict(rgb, BOX)


def test_animated_png_is_not_accepted_as_static_mask(rgb):
    with Image.new("L", rgb.size, 0) as first, Image.new("L", rgb.size, 255) as second, BytesIO() as stream:
        first.save(stream, format="PNG", save_all=True, append_images=[second], duration=100)
        instance, _ = backend(token(), {"image": base64.b64encode(stream.getvalue()).decode()})
    with pytest.raises(SegmentationAPIError, match="多帧掩码"):
        instance.predict(rgb, BOX)


def test_mask_decoded_byte_limit_is_enforced(monkeypatch, rgb):
    monkeypatch.setattr(api, "MAX_MASK_BYTES", 8)
    instance, _ = backend(token(), mask_response())
    with pytest.raises(SegmentationAPIError, match="掩码文件超过"):
        instance.predict(rgb, BOX)


def test_gray_mask_is_thresholded_with_explicit_warning(rgb):
    instance, _ = backend(token(), mask_response(gray=True))
    result = instance.predict(rgb, BOX)
    with result.mask:
        assert set(result.mask.tobytes()) == {0, 255}
        assert result.mask.getpixel((30, 40)) == 255
    assert any("128阈值" in item for item in result.warnings)


@pytest.mark.parametrize("fill", [0, 255])
def test_empty_and_full_masks_are_real_outputs_for_processor_review(rgb, fill):
    with Image.new("L", rgb.size, fill) as mask:
        instance, _ = backend(token(), {"image": png64(mask), "score": 0.99})
    result = instance.predict(rgb, BOX)
    with result.mask:
        assert result.mask.getextrema() == (fill, fill)
    assert result.score is None  # Provider does not document a confidence score.


@pytest.mark.parametrize("credentials", [("", SECRET), (KEY, ""), (" ", " ")])
def test_missing_credentials_are_unavailable_without_network(rgb, credentials):
    opener = Opener()
    instance = BaiduSegmentationBackend(*credentials, opener=opener)
    with pytest.raises(SegmentationUnavailable, match="尚未配置"):
        instance.predict(rgb, BOX)
    assert opener.calls == []


@pytest.mark.parametrize("timeout", [0, -1, 301, float("nan"), float("inf"), True, "30"])
def test_timeout_validation(timeout):
    with pytest.raises(ValueError):
        BaiduSegmentationBackend(KEY, SECRET, timeout)


@pytest.mark.parametrize("payload", [
    {"error": "invalid_client", "error_description": SECRET},
    {"access_token": "injected\r\nheader", "expires_in": 3600},
    {"access_token": TOKEN, "expires_in": True},
    {"access_token": TOKEN, "expires_in": 0},
    {"access_token": TOKEN, "expires_in": "3600"},
])
def test_invalid_token_response_is_not_logged_or_exposed(rgb, payload, caplog):
    instance, opener = backend(payload)
    with pytest.raises(SegmentationUnavailable) as caught:
        instance.predict(rgb, BOX)
    assert len(opener.calls) == 1
    evidence = str(caught.value) + caplog.text
    assert KEY not in evidence and SECRET not in evidence and TOKEN not in evidence


@pytest.mark.parametrize("code", [401, 403, 429, 500, 302])
def test_http_failures_are_safe_and_never_automatically_retried(rgb, code, caplog):
    private_url = api.SEGMENT_URL + "?access_token=" + TOKEN
    failure = HTTPError(private_url, code, SECRET, {}, BytesIO(SECRET.encode()))
    instance, opener = backend(token(), failure)
    expected = SegmentationUnavailable if code in {401, 403} else SegmentationAPIError
    with pytest.raises(expected) as caught:
        instance.predict(rgb, BOX)
    assert len(opener.calls) == 2
    evidence = "".join(traceback.format_exception(caught.value)) + caplog.text
    for private in (KEY, SECRET, TOKEN, private_url):
        assert private not in evidence
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize("failure", [TimeoutError(SECRET), URLError(TOKEN), OSError(KEY)])
def test_network_failures_do_not_leak_credentials_or_retry(rgb, failure, caplog):
    instance, opener = backend(token(), failure)
    with pytest.raises(SegmentationAPIError, match="未自动重试") as caught:
        instance.predict(rgb, BOX)
    assert len(opener.calls) == 2
    evidence = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert all(private not in evidence for private in (KEY, SECRET, TOKEN))


@pytest.mark.parametrize("error_code", [18, "bad", {}, [], True])
def test_api_error_codes_are_safely_normalized(rgb, error_code):
    instance, opener = backend(token(), {"error_code": error_code, "error_msg": SECRET})
    with pytest.raises(SegmentationAPIError) as caught:
        instance.predict(rgb, BOX)
    assert SECRET not in str(caught.value)
    assert len(opener.calls) == 2


def test_expired_token_clears_cache_but_does_not_repeat_paid_request(rgb):
    instance, opener = backend(token(), {"error_code": 110, "error_msg": SECRET}, token("new"), mask_response())
    with pytest.raises(SegmentationUnavailable, match="没有自动重试"):
        instance.predict(rgb, BOX)
    assert len(opener.calls) == 2
    instance.predict(rgb, BOX).mask.close()
    assert len(opener.calls) == 4
    assert parse_qs(urlsplit(opener.calls[3][0].full_url).query)["access_token"] == ["new"]


@pytest.mark.parametrize("body", [b"not-json", b"\xff", b"[]", b"null"])
def test_malformed_json_is_safe(rgb, body):
    instance, _ = backend(token(), Response(body))
    with pytest.raises(SegmentationAPIError):
        instance.predict(rgb, BOX)


def test_response_size_is_bounded(monkeypatch, rgb):
    monkeypatch.setattr(api, "MAX_RESPONSE_BYTES", 256)
    instance, _ = backend(token(), {"image": "A" * 257})
    with pytest.raises(SegmentationAPIError, match="响应超过"):
        instance.predict(rgb, BOX)


def test_token_response_size_is_bounded(monkeypatch, rgb):
    monkeypatch.setattr(api, "MAX_TOKEN_RESPONSE_BYTES", 64)
    instance, opener = backend({"access_token": TOKEN * 100, "expires_in": 3600})
    with pytest.raises(SegmentationAPIError, match="响应超过"):
        instance.predict(rgb, BOX)
    assert len(opener.calls) == 1


def test_redirect_handler_refuses_credential_forwarding():
    handler = api._NoRedirect()
    assert handler.redirect_request(None, None, 302, "moved", {}, "https://attacker.invalid/") is None
    default = BaiduSegmentationBackend(KEY, SECRET)
    assert any(isinstance(item, api._NoRedirect) for item in default._opener.handlers)


def test_typed_box_request_implements_backend_protocol_and_preserves_provider_identity(rgb, monkeypatch):
    ticks = iter([10.0, 10.375])
    monkeypatch.setattr(api.time, "perf_counter", lambda: next(ticks))
    instance, opener = backend(token(), {**mask_response(), "log_id": SECRET, "score": 0.99})
    assert isinstance(instance, SegmentationBackend)
    result = instance.predict(SegmentationRequest(rgb, BoxPrompt(BOX)))
    with result.mask:
        assert result.mask.mode == "L" and result.mask.size == rgb.size
        assert set(result.mask.tobytes()) == {0, 255}
    assert result.model_id == "baidu"
    assert result.model_version == "service-unversioned"
    assert result.processing_time == pytest.approx(0.375)
    assert result.score is None and result.score_type is None
    assert result.candidates == ()
    assert result.extensions == {"baidu": {"api_mode": "control"}}
    evidence = repr(result.extensions)
    assert all(value not in evidence for value in (KEY, SECRET, TOKEN))
    payload = json.loads(opener.calls[1][0].data)
    assert payload["position"] == [[[16, 20], [220, 170]]]


def test_automatic_request_uses_official_auto_mode_without_invented_box(rgb):
    instance, opener = backend(token(), mask_response())
    result = instance.predict(SegmentationRequest(rgb, AutomaticPrompt()))
    with result.mask:
        assert result.mask.mode == "L" and result.mask.size == rgb.size
        assert result.mask.getpixel((30, 40)) == 255
    assert instance.capabilities.automatic_mask is True
    assert instance.capabilities.box_prompt is True
    assert instance.capabilities.point_prompt is False
    assert instance.capabilities.quality_score is False
    assert result.score is None and result.score_type is None
    assert result.extensions == {"baidu": {"api_mode": "auto"}}
    payload = json.loads(opener.calls[1][0].data)
    assert payload["method"] == "auto"
    assert "position" not in payload
    assert set(payload) == {"image", "method", "return_form", "refine_mask"}


def test_automatic_mask_maps_through_padding_without_mutating_original():
    with Image.new("L", (128, 128)) as returned:
        ImageDraw.Draw(returned).rectangle((48, 52, 79, 75), fill=255)
        instance, opener = backend(token(), {"image": png64(returned)})
    with Image.new("RGB", (32, 24), (17, 43, 111)) as image:
        before = image.tobytes()
        result = instance.predict(SegmentationRequest(image, AutomaticPrompt()))
        with result.mask:
            assert result.mask.size == image.size and result.mask.getextrema() == (255, 255)
        assert image.tobytes() == before
    assert any("白色边距" in item for item in result.warnings)
    assert not any("提示框" in item for item in result.warnings)
    assert "position" not in json.loads(opener.calls[1][0].data)


@pytest.mark.parametrize("prompt_factory", [
    lambda size: PointPrompt(points=((20.0, 30.0),), labels=(1,)),
    lambda size: TextPrompt("杯子"),
    lambda size: MaskPrompt(Image.new("L", size, 0)),
])
def test_unimplemented_provider_prompt_is_rejected_before_credentials_or_network(rgb, prompt_factory):
    prompt = prompt_factory(rgb.size)
    opener = Opener()
    instance = BaiduSegmentationBackend("", "", opener=opener)
    try:
        with pytest.raises(UnsupportedPromptError):
            instance.predict(SegmentationRequest(rgb, prompt))
        assert opener.calls == []
    finally:
        if isinstance(prompt, MaskPrompt):
            prompt.mask.close()


def test_legacy_call_without_box_is_not_silently_changed_to_automatic(rgb):
    instance, opener = backend()
    with pytest.raises(SegmentationAPIError):
        instance.predict(rgb)
    assert opener.calls == []


def test_typed_request_cannot_also_supply_a_legacy_box(rgb):
    instance, opener = backend()
    with pytest.raises(SegmentationAPIError):
        instance.predict(SegmentationRequest(rgb, AutomaticPrompt()), BOX)
    assert opener.calls == []
