from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps.model_versions import get_model_version_service_dep
from api.modules.model_versions.schemas import (
    ModelVersionCreateRequest,
    ModelVersionListResponse,
    ModelVersionResponse,
)
from api.modules.model_versions.service import ModelVersionService

router = APIRouter(prefix="/model-versions", tags=["model_versions"])
MODEL_VERSION_SERVICE_DEP = Depends(get_model_version_service_dep)


@router.post(
    "",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Model Version",
    description="Creates a model version metadata entry.",
    responses={
        201: {
            "description": "Model version created.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "fa871058-6b3f-40a4-b3d2-052ab08a1985",
                        "name": "transformer-v7",
                        "source_checkpoint": "model_iter_006.pt",
                        "is_active": False,
                        "notes": "Kaggle run week 7",
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        409: {
            "description": "Model version name already exists.",
            "content": {
                "application/json": {
                    "example": {"detail": "Model version name already exists: transformer-v7"},
                }
            },
        },
    },
)
async def post_model_version(
    request: ModelVersionCreateRequest,
    service: ModelVersionService = MODEL_VERSION_SERVICE_DEP,
) -> ModelVersionResponse:
    try:
        version = await service.create_model_version(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return ModelVersionResponse.model_validate(version)


@router.get(
    "",
    response_model=ModelVersionListResponse,
    summary="List Model Versions",
    description="Returns model version entries with optional limit.",
    responses={
        200: {
            "description": "Model versions returned.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "fa871058-6b3f-40a4-b3d2-052ab08a1985",
                            "name": "transformer-v7",
                            "source_checkpoint": "model_iter_006.pt",
                            "is_active": True,
                            "notes": "Primary production model",
                            "created_at": "2026-02-22T12:00:00Z",
                        }
                    ]
                }
            },
        }
    },
)
async def list_model_versions(
    limit: int = 50,
    offset: int = 0,
    service: ModelVersionService = MODEL_VERSION_SERVICE_DEP,
) -> ModelVersionListResponse:
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    total, versions = await service.list_model_versions(limit=safe_limit, offset=safe_offset)
    items = [ModelVersionResponse.model_validate(row) for row in versions]
    return ModelVersionListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


@router.get(
    "/active",
    response_model=ModelVersionResponse,
    summary="Get Active Model Version",
    description="Returns the currently active model version.",
    responses={
        200: {
            "description": "Active model version returned.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "fa871058-6b3f-40a4-b3d2-052ab08a1985",
                        "name": "transformer-v7",
                        "source_checkpoint": "model_iter_006.pt",
                        "is_active": True,
                        "notes": "Primary production model",
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        404: {
            "description": "No active model version found.",
            "content": {
                "application/json": {
                    "example": {"detail": "No active model version found."},
                }
            },
        },
    },
)
async def get_active_model_version(
    service: ModelVersionService = MODEL_VERSION_SERVICE_DEP,
) -> ModelVersionResponse:
    version = await service.get_active_model_version()
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active model version found.",
        )
    return ModelVersionResponse.model_validate(version)


@router.get(
    "/{version_id}",
    response_model=ModelVersionResponse,
    summary="Get Model Version",
    description="Returns model version by ID.",
    responses={
        200: {
            "description": "Model version returned.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "fa871058-6b3f-40a4-b3d2-052ab08a1985",
                        "name": "transformer-v7",
                        "source_checkpoint": "model_iter_006.pt",
                        "is_active": False,
                        "notes": "Archived baseline",
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        404: {
            "description": "Model version not found.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model version not found: fa871058-6b3f-40a4-b3d2-052ab08a1985",
                    }
                }
            },
        },
    },
)
async def get_model_version(
    version_id: UUID,
    service: ModelVersionService = MODEL_VERSION_SERVICE_DEP,
) -> ModelVersionResponse:
    version = await service.get_model_version(version_id)
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version not found: {version_id}",
        )
    return ModelVersionResponse.model_validate(version)


@router.post(
    "/{version_id}/activate",
    response_model=ModelVersionResponse,
    summary="Activate Model Version",
    description="Activates one model version and deactivates others.",
    responses={
        200: {
            "description": "Model version activated.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "fa871058-6b3f-40a4-b3d2-052ab08a1985",
                        "name": "transformer-v7",
                        "source_checkpoint": "model_iter_006.pt",
                        "is_active": True,
                        "notes": "Now active model",
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        404: {
            "description": "Model version not found.",
            "content": {
                "application/json": {
                    "example": {"detail": "Model version not found: fa871058-6b3f-40a4-b3d2-052ab08a1985"},
                }
            },
        },
    },
)
async def post_activate_model_version(
    version_id: UUID,
    service: ModelVersionService = MODEL_VERSION_SERVICE_DEP,
) -> ModelVersionResponse:
    try:
        version = await service.activate_model_version(version_id=version_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return ModelVersionResponse.model_validate(version)
