"""Thin HTTP layer: receive input, call the application service, return saved data."""

import logging
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from app.config import PROJECT_ROOT, Settings
from app.models.object_profile import ObjectProfile
from app.services.image_collection import CollectionError, ImageInput
from app.storage.local_storage import CorruptProfileError, ProfileNotFoundError


logger = logging.getLogger(__name__)
router = APIRouter()
BODY_LIMIT_MARKER = "digitizer_request_body_limit"


class LocalMultipartParser(MultiPartParser):
    """Keep the multipart parser's disk spools inside the configured data directory."""

    def __init__(self, *args, spool_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.spool_dir = spool_dir
        self._complete = False

    def on_end(self) -> None:
        super().on_end()
        self._complete = True

    async def parse(self) -> FormData:
        try:
            form = await super().parse()
            # python-multipart may finish a truncated stream without raising.
            # Only its final-boundary callback confirms the whole batch arrived.
            if not self._complete:
                raise MultiPartException("Incomplete multipart data.")
            return form
        except BaseException:
            # An unfinished last part is absent from FormData, so close every
            # allocated spool, including files which have not reached part-end.
            for spool in self._files_to_close_on_error:
                spool.close()
            raise

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        if self._current_part.file is not None:
            # Starlette has just allocated an empty in-memory spool. Replace it
            # before receiving any file data, retaining its normal parser checks.
            original = self._current_part.file.file
            original.close()
            local_spool = SpooledTemporaryFile(max_size=self.spool_max_size, mode="w+b", dir=self.spool_dir)
            self._current_part.file.file = local_spool
            self._files_to_close_on_error.append(local_spool)


class CreateObjectResponse(BaseModel):
    object_id: UUID
    status: Literal["completed", "completed_with_warnings"]
    profile: ObjectProfile


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app" / "web" / "index.html")


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@router.get("/api/v1/config")
def public_config(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    return {"max_files": settings.max_files, "max_image_bytes": settings.max_image_bytes,
            "max_total_bytes": settings.max_total_bytes, "max_image_pixels": settings.max_image_pixels}


@router.post("/api/v1/objects", status_code=201, response_model=CreateObjectResponse,
             openapi_extra={"requestBody": {"required": True, "content": {
                 "multipart/form-data": {"schema": {"type": "object", "required": ["name", "images"],
                     "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 120},
                                    "images": {"type": "array", "items": {"type": "string", "format": "binary"}}}}}}}})
async def create_object(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise api_error(422, "invalid_upload_form", "请使用 multipart/form-data 同时提交物体名称和图片。")
    spool_dir = settings.data_dir / ".multipart"
    await run_in_threadpool(spool_dir.mkdir, parents=True, exist_ok=True)
    try:
        parser = LocalMultipartParser(request.headers, request.stream(), spool_dir=spool_dir,
                                      max_files=settings.max_files, max_fields=1, max_part_size=4096)
        form = await parser.parse()
    except MultiPartException as exc:
        if exc.message == BODY_LIMIT_MARKER:
            raise api_error(413, "upload_too_large", "上传请求超过总大小限制。") from exc
        raise api_error(422, "invalid_upload_form", "上传表单无效、字段过多或图片数量超过限制。") from exc
    try:
        names = form.getlist("name")
        uploads = form.getlist("images")
        if set(form.keys()) - {"name", "images"}:
            raise api_error(422, "invalid_upload_form", "表单只接受物体名称和图片。")
        if len(names) != 1 or not isinstance(names[0], str):
            raise api_error(422, "invalid_name", "请填写一个物体名称。")
        if any(not isinstance(upload, UploadFile) for upload in uploads):
            raise api_error(422, "invalid_images", f"请选择 1–{settings.max_files} 张图片。")
        images = [ImageInput(upload.filename or "image", upload.file) for upload in uploads]
        try:
            profile = await run_in_threadpool(request.app.state.collection_service.collect, names[0], images)
        except CollectionError as exc:
            raise api_error(exc.status_code, exc.code, str(exc)) from exc
        return {"object_id": str(profile.object_id), "status": profile.status,
                "profile": profile.model_dump(mode="json")}
    finally:
        await form.close()


async def _read_profile(request: Request, object_id: UUID) -> tuple[ObjectProfile, bytes]:
    try:
        return await run_in_threadpool(request.app.state.storage.read_profile, object_id)
    except ProfileNotFoundError as exc:
        raise api_error(404, "object_not_found", "未找到该物体档案。") from exc
    except CorruptProfileError as exc:
        logger.exception("Stored profile validation failed for object %s", object_id)
        raise api_error(500, "profile_unavailable", "档案读取校验失败，请查看服务日志。") from exc


@router.get("/api/v1/objects/{object_id}", response_model=ObjectProfile)
async def get_object(request: Request, object_id: UUID) -> ObjectProfile:
    profile, _ = await _read_profile(request, object_id)
    return profile


@router.get("/api/v1/objects/{object_id}/profile.json")
async def download_profile(request: Request, object_id: UUID) -> Response:
    _, content = await _read_profile(request, object_id)
    return Response(content, media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="{object_id}-profile.json"',
        "X-Content-Type-Options": "nosniff",
    })
