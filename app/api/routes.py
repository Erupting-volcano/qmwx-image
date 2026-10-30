"""Thin HTTP layer: receive input, call the application service, return saved data."""

import json
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
from app.models.segmentation import MODEL_NAME, NormalizedBox, SegmentationData
from app.services.image_collection import CollectionError, ImageInput
from app.storage.local_storage import CorruptProfileError, ProfileNotFoundError, resolve_mask_path, resolve_original_path


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
            "max_total_bytes": settings.max_total_bytes, "max_image_pixels": settings.max_image_pixels,
            "segmentation_enabled": settings.segmentation_enabled,
            "segmentation_prompt_type": "box", "segmentation_model": MODEL_NAME,
            "segmentation_provider": "百度智能云智能抠图"}


def _parse_segmentation_boxes(values: list, image_count: int) -> list[NormalizedBox | None]:
    if not values:
        return [None] * image_count
    try:
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("Expected one JSON field")
        payload = json.loads(values[0])
        if not isinstance(payload, list) or len(payload) != image_count:
            raise ValueError("Image count mismatch")
        boxes = []
        for value in payload:
            if value is None:
                boxes.append(None)
            else:
                if not isinstance(value, list) or len(value) != 4:
                    raise ValueError("Expected four coordinates")
                boxes.append(NormalizedBox(x0=value[0], y0=value[1], x1=value[2], y1=value[3]))
        return boxes
    except (ValueError, TypeError) as exc:
        raise api_error(422, "invalid_segmentation_boxes",
                        "目标框需与图片顺序一一对应，使用 null 或 [左, 上, 右, 下] 的有效 0–1 比例坐标。") from exc


@router.post("/api/v1/objects", status_code=201, response_model=CreateObjectResponse,
             openapi_extra={
                 "requestBody": {"required": True, "content": {
                     "multipart/form-data": {"schema": {
                         "type": "object", "required": ["name", "images"],
                         "properties": {
                             "name": {"type": "string", "minLength": 1, "maxLength": 120},
                             "images": {"type": "array", "items": {"type": "string", "format": "binary"}},
                             "segmentation_boxes": {"type": "string", "description":
                                 "可选JSON数组，与图片顺序一致，每项null或EXIF转正后[x0,y0,x1,y1]比例坐标"},
                         },
                     }},
                 }},
             })
async def create_object(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise api_error(422, "invalid_upload_form", "请使用 multipart/form-data 同时提交物体名称和图片。")
    spool_dir = settings.data_dir / ".multipart"
    await run_in_threadpool(spool_dir.mkdir, parents=True, exist_ok=True)
    try:
        parser = LocalMultipartParser(request.headers, request.stream(), spool_dir=spool_dir,
                                      max_files=settings.max_files, max_fields=2, max_part_size=4096)
        form = await parser.parse()
    except MultiPartException as exc:
        if exc.message == BODY_LIMIT_MARKER:
            raise api_error(413, "upload_too_large", "上传请求超过总大小限制。") from exc
        raise api_error(422, "invalid_upload_form", "上传表单无效、字段过多或图片数量超过限制。") from exc
    try:
        names = form.getlist("name")
        uploads = form.getlist("images")
        if set(form.keys()) - {"name", "images", "segmentation_boxes"}:
            raise api_error(422, "invalid_upload_form", "表单只接受物体名称、图片和可选目标框。")
        if len(names) != 1 or not isinstance(names[0], str):
            raise api_error(422, "invalid_name", "请填写一个物体名称。")
        if any(not isinstance(upload, UploadFile) for upload in uploads):
            raise api_error(422, "invalid_images", f"请选择 1–{settings.max_files} 张图片。")
        boxes = _parse_segmentation_boxes(form.getlist("segmentation_boxes"), len(uploads))
        images = [ImageInput(upload.filename or "image", upload.file, box) for upload, box in zip(uploads, boxes)]
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


@router.get("/api/v1/objects/{object_id}/images/{asset_id}/original")
async def preview_original(request: Request, object_id: UUID, asset_id: UUID) -> FileResponse:
    profile, _ = await _read_profile(request, object_id)
    asset = next((image for image in profile.images if image.asset_id == asset_id), None)
    if asset is None:
        raise api_error(404, "image_not_found", "此物体档案中没有该图片。")
    try:
        path = await run_in_threadpool(resolve_original_path, request.app.state.storage.data_dir,
                                       object_id, asset_id, asset.storage_path)
    except (ValueError, TypeError, OSError) as exc:
        raise api_error(404, "image_not_found", "图片文件不可用。") from exc
    return FileResponse(path, media_type=asset.mime_type,
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})


@router.get("/api/v1/objects/{object_id}/images/{asset_id}/mask")
async def preview_mask(request: Request, object_id: UUID, asset_id: UUID) -> FileResponse:
    profile, _ = await _read_profile(request, object_id)
    if not any(image.asset_id == asset_id for image in profile.images):
        raise api_error(404, "mask_not_found", "此物体档案中没有该图片。")
    for result in reversed(profile.processing_results):
        if result.module_name != "object_segmentation":
            continue
        if result.status not in {"success", "partial"}:
            break
        try:
            payload = SegmentationData.model_validate(result.data)
            record = next((image for image in payload.images if image.asset_id == asset_id), None)
            if record is None or record.mask_path is None or record.status not in {"success", "needs_confirmation"}:
                break
            path = await run_in_threadpool(
                resolve_mask_path, request.app.state.storage.data_dir, object_id, asset_id, record.mask_path,
                expected_size=(record.width, record.height), max_pixels=request.app.state.settings.max_image_pixels,
                expected_bbox=record.bbox, expected_area_ratio=record.area_ratio,
            )
        except (ValueError, TypeError, OSError):
            break
        return FileResponse(path, media_type="image/png",
                            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})
    raise api_error(404, "mask_not_found", "此图片没有可预览的有效分割掩码。")
