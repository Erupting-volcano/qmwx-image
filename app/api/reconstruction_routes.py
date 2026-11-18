"""Thin HTTP surface for reconstruction jobs and their downloadable artifacts."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.models.object_profile import ObjectProfile
from app.models.reconstruction import ReconstructionRequest
from app.services.reconstruction_jobs import JobNotFoundError, JobStoreError
from app.services.reconstruction_service import ReconstructionUnavailable
from app.storage.local_storage import (
    CorruptProfileError, ProfileNotFoundError, resolve_artifact_path,
)


logger = logging.getLogger(__name__)
router = APIRouter()

ARTIFACT_FIELDS = {"point_cloud": "point_cloud_path", "mesh": "mesh_path"}


class CreateReconstructionBody(BaseModel):
    """Optional narrowing; omitting asset_ids means every image in the archive."""

    asset_ids: list[UUID] | None = None
    backend: str | None = None
    use_masks: bool = True


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _service(request: Request):
    service = getattr(request.app.state, "reconstruction_service", None)
    if service is None:
        raise api_error(503, "reconstruction_disabled", "三维重建未启用，请先在服务端开启并配置 COLMAP。")
    return service


@router.post("/api/v1/objects/{object_id}/reconstructions", status_code=202)
async def create_reconstruction(request: Request, object_id: UUID,
                                body: CreateReconstructionBody | None = None) -> dict:
    service = _service(request)
    body = body or CreateReconstructionBody()
    try:
        profile, _ = await run_in_threadpool(request.app.state.storage.read_profile, object_id)
    except ProfileNotFoundError as exc:
        raise api_error(404, "object_not_found", "未找到该物体档案。") from exc
    except CorruptProfileError as exc:
        raise api_error(500, "profile_unavailable", "档案读取校验失败，请查看服务日志。") from exc

    asset_ids = body.asset_ids or [image.asset_id for image in profile.images]
    backend_id = body.backend or service.default_backend_id
    if backend_id is None:
        raise api_error(503, "reconstruction_disabled", "没有可用的重建后端。")
    payload = ReconstructionRequest(object_id=object_id, asset_ids=asset_ids,
                                    backend_id=backend_id, use_masks=body.use_masks)
    try:
        job = await run_in_threadpool(service.submit, payload)
    except ReconstructionUnavailable as exc:
        raise api_error(409, "reconstruction_unavailable", str(exc)) from exc
    except ProfileNotFoundError as exc:
        raise api_error(404, "object_not_found", "未找到该物体档案。") from exc
    return {"job_id": str(job.job_id), "status": job.status}


@router.get("/api/v1/reconstructions/{job_id}")
async def get_reconstruction(request: Request, job_id: UUID) -> dict:
    service = _service(request)
    try:
        job = await run_in_threadpool(service.get, job_id)
    except JobNotFoundError as exc:
        raise api_error(404, "job_not_found", "未找到该重建任务。") from exc
    except JobStoreError as exc:
        raise api_error(500, "job_unavailable", "任务记录不可用，请查看服务日志。") from exc
    return job.model_dump(mode="json")


@router.post("/api/v1/reconstructions/{job_id}/cancel")
async def cancel_reconstruction(request: Request, job_id: UUID) -> dict:
    service = _service(request)
    try:
        job = await run_in_threadpool(service.cancel, job_id)
    except JobNotFoundError as exc:
        raise api_error(404, "job_not_found", "未找到该重建任务。") from exc
    return job.model_dump(mode="json")


@router.get("/api/v1/objects/{object_id}/geometry/{kind}")
async def download_geometry(request: Request, object_id: UUID, kind: str) -> FileResponse:
    field = ARTIFACT_FIELDS.get(kind)
    if field is None:
        raise api_error(404, "artifact_not_found", "只支持下载点云或网格。")
    settings = request.app.state.settings
    try:
        profile, _ = await run_in_threadpool(request.app.state.storage.read_profile, object_id)
    except ProfileNotFoundError as exc:
        raise api_error(404, "object_not_found", "未找到该物体档案。") from exc
    except CorruptProfileError as exc:
        raise api_error(500, "profile_unavailable", "档案读取校验失败，请查看服务日志。") from exc
    relative = getattr(profile.geometry, field)
    if not relative:
        raise api_error(404, "artifact_not_found", "该物体的档案中还没有这类几何产物。")
    try:
        path = await run_in_threadpool(
            resolve_artifact_path, request.app.state.storage.data_dir, object_id, relative,
            max_bytes=settings.reconstruction_max_artifact_bytes, expected_kind=kind,
        )
    except (ValueError, TypeError, OSError) as exc:
        logger.warning("Geometry artifact rejected for object %s: %s", object_id, exc)
        raise api_error(404, "artifact_not_found", "几何产物不可用。") from exc
    return FileResponse(path, media_type="application/octet-stream", filename=f"{object_id}-{kind}{path.suffix}",
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})
