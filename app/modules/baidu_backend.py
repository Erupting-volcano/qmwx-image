"""Baidu intelligent segmentation over HTTPS; no local neural model required.

Official request and response contract:
https://cloud.baidu.com/doc/IMAGEPROCESS/s/rm8zl3koj
Official token form POST example:
https://cloud.baidu.com/doc/IMAGEPROCESS/s/2k3bclou0
"""

import base64
import binascii
from io import BytesIO
import json
import math
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, UnidentifiedImageError

from app.modules.segmentation_backend import MaskPrediction, SegmentationAPIError, SegmentationUnavailable


MODEL_NAME = "baidu_intelligent_segmentation"
TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
SEGMENT_URL = "https://aip.baidubce.com/rest/2.0/image-process/v1/segment"
MIN_API_SIDE = 128
MAX_API_SIDE = 3000
MAX_IMAGE_BASE64_BYTES = 9_500_000  # Below the documented 10 MB limit.
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_TOKEN_RESPONSE_BYTES = 64 * 1024
MAX_MASK_BYTES = 16 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentials and image bodies must never follow a provider-supplied URL.
        return None


class BaiduSegmentationBackend:
    model_name = MODEL_NAME

    def __init__(self, api_key: str, secret_key: str, timeout_seconds: float = 30, *, opener=None) -> None:
        if not isinstance(api_key, str) or not isinstance(secret_key, str):
            raise ValueError("API凭证必须为字符串")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300):
            raise ValueError("API超时必须为0–300秒之间的有限正数")
        self._api_key = api_key.strip()
        self._secret_key = secret_key.strip()
        self.timeout_seconds = timeout_seconds
        self._opener = opener if opener is not None else build_opener(_NoRedirect())
        self._token: str | None = None
        self._expires_at = 0.0
        self._token_lock = Lock()

    @staticmethod
    def _http_failure(status: int) -> Exception:
        if status in {401, 403}:
            return SegmentationUnavailable("百度分割API鉴权或授权失败，请检查应用密钥及智能抠图权限。")
        if status == 429:
            return SegmentationAPIError("百度分割API请求过于频繁或额度受限，请检查额度后稍后重试；本次未自动重试。")
        if 300 <= status < 400:
            return SegmentationAPIError("百度分割API返回了重定向，为保护图片和凭证已拒绝跟随。")
        if status >= 500:
            return SegmentationAPIError("百度分割API服务暂时不可用，请稍后重试；本次未自动重试。")
        return SegmentationAPIError("百度分割API拒绝了本次请求，请检查服务权限和图片参数。")

    def _request_json(self, request: Request, max_bytes: int) -> dict:
        # urllib does not log credential-bearing URLs. Never expose exception text,
        # response bodies, or chained transport exceptions to the processor logger.
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if not isinstance(status, int) or not 200 <= status < 300:
                    raise self._http_failure(status if isinstance(status, int) else 400)
                content = response.read(max_bytes + 1)
                if len(content) > max_bytes:
                    raise SegmentationAPIError("百度分割API响应超过安全大小限制。")
            payload = json.loads(content.decode("utf-8"))
            if not isinstance(payload, dict):
                raise SegmentationAPIError("百度分割API返回的数据格式无效。")
            return payload
        except HTTPError as error:
            status = error.code
            error.close()
            raise self._http_failure(status) from None
        except (TimeoutError, URLError, OSError):
            raise SegmentationAPIError("百度分割API连接失败或请求超时，请检查网络后重试；本次未自动重试。") from None
        except (UnicodeError, json.JSONDecodeError):
            raise SegmentationAPIError("百度分割API返回的数据无法解析。") from None
        except (SegmentationUnavailable, SegmentationAPIError):
            raise
        except Exception:
            raise SegmentationAPIError("百度分割API请求未完成，请检查网络和服务状态。") from None

    def _invalidate_token(self, token: str) -> None:
        with self._token_lock:
            if self._token == token:
                self._token = None
                self._expires_at = 0.0

    def _access_token(self) -> str:
        if not self._api_key or not self._secret_key:
            raise SegmentationUnavailable("尚未配置百度API Key和Secret Key，请配置后重启服务；不会使用本地模型或伪造掩码。")
        cached_token = self._token
        if cached_token and time.monotonic() < self._expires_at:
            return cached_token
        with self._token_lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            body = urlencode({"grant_type": "client_credentials", "client_id": self._api_key,
                              "client_secret": self._secret_key}).encode("ascii")
            payload = self._request_json(Request(
                TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
            ), MAX_TOKEN_RESPONSE_BYTES)
            token = payload.get("access_token")
            expires = payload.get("expires_in")
            if (payload.get("error") or not isinstance(token, str) or not token or len(token) > 4096
                    or any(ord(char) < 33 or ord(char) > 126 for char in token)
                    or isinstance(expires, bool) or not isinstance(expires, (int, float))
                    or not math.isfinite(expires) or expires <= 0):
                raise SegmentationUnavailable("百度API令牌获取失败，请检查应用API Key、Secret Key和服务开通状态。")
            self._token = token
            self._expires_at = time.monotonic() + max(0.0, expires - 60)
            return token

    @staticmethod
    def _prepare_image(image: Image.Image, box: tuple[float, float, float, float]):
        if not isinstance(image, Image.Image) or image.mode != "RGB" or min(image.size) <= 0:
            raise SegmentationAPIError("分割API需要已转正的有效RGB图片。")
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box)):
            raise SegmentationAPIError("目标框坐标无效，请重新框选。")
        x0, y0, x1, y1 = box
        width, height = image.size
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise SegmentationAPIError("目标框超出图片范围，请重新框选。")
        scale = min(1.0, MAX_API_SIDE / max(width, height))
        scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        api_size = (max(MIN_API_SIDE, scaled_size[0]), max(MIN_API_SIDE, scaled_size[1]))
        left = (api_size[0] - scaled_size[0]) // 2
        top = (api_size[1] - scaled_size[1]) // 2
        transformed = (math.floor(x0 * scaled_size[0] / width + left),
                       math.floor(y0 * scaled_size[1] / height + top),
                       math.ceil(x1 * scaled_size[0] / width + left),
                       math.ceil(y1 * scaled_size[1] / height + top))
        api_box = (max(1, transformed[0]), max(1, transformed[1]),
                   min(api_size[0] - 1, transformed[2]), min(api_size[1] - 1, transformed[3]))
        if api_box[2] - api_box[0] < 10 or api_box[3] - api_box[1] < 10:
            raise SegmentationAPIError("目标框映射到API图片后小于10×10像素，请框住更完整的物体或补拍近景。")
        warnings = []
        if scaled_size != image.size:
            warnings.append("图片已等比缩小至API允许尺寸，掩码将映射回原图；边界精度受缩小分辨率影响。")
        if api_size != scaled_size:
            warnings.append("为满足API最短边128像素的要求，发送副本增加了白色边距，返回时会移除边距。")
        if api_box != transformed:
            warnings.append("API不允许目标框贴边，已将接口提示框向内调整1像素，请检查靠近图边的物体边界。")
        prepared = Image.new("RGB", api_size, "white")
        if scaled_size != image.size:
            with image.resize(scaled_size, Image.Resampling.LANCZOS) as reduced:
                prepared.paste(reduced, (left, top))
        else:
            prepared.paste(image, (left, top))
        region = (left, top, left + scaled_size[0], top + scaled_size[1])
        return prepared, api_box, region, warnings

    @staticmethod
    def _encode_image(image: Image.Image, warnings: list[str]) -> str:
        stream = BytesIO()
        image.save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        if len(encoded) <= MAX_IMAGE_BASE64_BYTES:
            return encoded
        for quality in (90, 80, 70):
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=quality)
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            if len(encoded) <= MAX_IMAGE_BASE64_BYTES:
                warnings.append("为满足API请求大小限制，发送了JPEG压缩副本；原图未修改，细边界可能受压缩影响。")
                return encoded
        raise SegmentationAPIError("图片编码后超过API安全大小限制，请减小图片或重新拍摄。")

    @staticmethod
    def _decode_mask(encoded, expected_size: tuple[int, int], warnings: list[str]) -> Image.Image:
        if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_RESPONSE_BYTES:
            raise SegmentationAPIError("百度分割API没有返回有效掩码。")
        try:
            content = base64.b64decode(encoded, validate=True)
            if not content or len(content) > MAX_MASK_BYTES:
                raise SegmentationAPIError("API掩码文件超过安全大小限制。")
            with Image.open(BytesIO(content)) as returned:
                if (returned.format != "PNG" or returned.mode not in {"L", "1"}
                        or returned.size != expected_size or getattr(returned, "n_frames", 1) != 1):
                    raise SegmentationAPIError("API掩码的格式、模式或尺寸与送入API的图片不一致。")
                returned.load()
                mask = returned.convert("L")
            histogram = mask.histogram()
            if sum(histogram[1:255]):
                with mask:
                    binary = mask.point(lambda value: 255 if value >= 128 else 0)
                warnings.append("API返回了灰度边缘，已按128阈值二值化；请人工复查细边界。")
                mask = binary
            return mask
        except (binascii.Error, ValueError, UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
            raise SegmentationAPIError("百度分割API返回的掩码无法安全解码。") from None

    def predict(self, image: Image.Image, box: tuple[float, float, float, float]) -> MaskPrediction:
        if not self._api_key or not self._secret_key:
            raise SegmentationUnavailable("尚未配置百度API Key和Secret Key，请配置后重启服务；不会使用本地模型或伪造掩码。")
        prepared, api_box, region, warnings = self._prepare_image(image, box)
        with prepared:
            encoded = self._encode_image(prepared, warnings)
            api_size = prepared.size
        token = self._access_token()
        body = json.dumps({"image": encoded, "method": "control",
                           "position": [[[api_box[0], api_box[1]], [api_box[2], api_box[3]]]],
                           "return_form": "mask", "refine_mask": "false"}, separators=(",", ":")).encode("utf-8")
        request = Request(SEGMENT_URL + "?" + urlencode({"access_token": token}), data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            payload = self._request_json(request, MAX_RESPONSE_BYTES)
        except SegmentationUnavailable:
            self._invalidate_token(token)
            raise
        error_code = payload.get("error_code")
        valid_code = isinstance(error_code, int) and not isinstance(error_code, bool)
        if error_code is not None and not (valid_code and error_code == 0):
            if valid_code and error_code in {110, 111}:
                self._invalidate_token(token)
                raise SegmentationUnavailable("百度访问令牌无效或已过期，请重试以重新获取令牌；本次没有自动重试分割请求。")
            code = str(error_code) if isinstance(error_code, int) and not isinstance(error_code, bool) else "未知"
            raise SegmentationAPIError(f"百度分割API返回错误（错误码{code}），请检查服务权限、额度和图片参数。")
        with self._decode_mask(payload.get("image"), api_size, warnings) as api_mask:
            with api_mask.crop(region) as unpadded:
                mask = unpadded.resize(image.size, Image.Resampling.NEAREST) if unpadded.size != image.size else unpadded.copy()
        return MaskPrediction(mask=mask, score=None, warnings=tuple(warnings))
