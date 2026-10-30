"""Run with: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000."""

from collections.abc import Sequence
import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from app.api.routes import BODY_LIMIT_MARKER, router
from app.config import PROJECT_ROOT, Settings
from app.modules.base import BaseProcessor
from app.services.pipeline import Pipeline
from app.services.image_collection import ImageCollectionService


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class UploadBodyLimitMiddleware:
    """Bound the incoming multipart body before it can fill temporary storage."""

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/v1/objects":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared_size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared_size = 0
        if declared_size > self.limit:
            response = JSONResponse(status_code=413, content={"detail": {
                "code": "upload_too_large", "message": "上传请求超过总大小限制。"}})
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    # The multipart parser closes its spool files on MultiPartException.
                    raise MultiPartException(BODY_LIMIT_MARKER)
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None,
               processors: Sequence[BaseProcessor] | None = None,
               *, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or Settings()
    collection_service = ImageCollectionService(settings, processors, pipeline=pipeline)
    application = FastAPI(title="其命维新 · 物体信息输入与数字化模块", version="0.1.0",
                          description="旧物改造主项目的基础模块：图片输入、元数据、质量筛查及可选的人工框选前景分割。")
    application.state.settings = settings
    application.state.storage = collection_service.storage
    application.state.collection_service = collection_service
    application.add_middleware(UploadBodyLimitMiddleware,
                               limit=settings.max_total_bytes + 1024 * 1024)
    application.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "web"), name="static")
    application.include_router(router)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": {
            "code": "invalid_request", "message": "请求参数不符合要求，请检查名称、图片和物体 ID。"}})

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "code": "http_error", "message": "请求的资源不存在或当前请求不可用。"}
        return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        error_id = str(uuid4())
        logger.error("Request failed, error_id=%s", error_id, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": {
            "code": "internal_error", "message": "处理或保存失败，请重试或查看服务日志。", "error_id": error_id}})

    return application


app = create_app()
