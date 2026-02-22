from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelVersionCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "ataxx-transformer-v7",
                "hf_repo_id": "dieg0code/ataxx-zero",
                "hf_revision": "main",
                "checkpoint_uri": "hf://dieg0code/ataxx-zero/model_iter_030.pt",
                "onnx_uri": "hf://dieg0code/ataxx-zero/model_iter_030.onnx",
                "is_active": True,
                "notes": "Best checkpoint from kaggle run 2026-02-20",
            }
        }
    )

    name: str = Field(min_length=1, max_length=120)
    hf_repo_id: str | None = Field(default=None, max_length=255)
    hf_revision: str | None = Field(default=None, max_length=255)
    checkpoint_uri: str | None = Field(default=None, max_length=500)
    onnx_uri: str | None = Field(default=None, max_length=500)
    is_active: bool = False
    notes: str | None = Field(default=None, max_length=2000)


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "1932ac4a-5dcf-4dc9-8f99-fdf7ef20cc99",
                "name": "ataxx-transformer-v7",
                "hf_repo_id": "dieg0code/ataxx-zero",
                "hf_revision": "main",
                "checkpoint_uri": "hf://dieg0code/ataxx-zero/model_iter_030.pt",
                "onnx_uri": "hf://dieg0code/ataxx-zero/model_iter_030.onnx",
                "is_active": True,
                "notes": "Best checkpoint from kaggle run 2026-02-20",
                "created_at": "2026-02-22T21:00:00+00:00",
            }
        },
    )

    id: UUID
    name: str
    hf_repo_id: str | None
    hf_revision: str | None
    checkpoint_uri: str | None
    onnx_uri: str | None
    is_active: bool
    notes: str | None
    created_at: datetime


class ModelVersionListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 0,
                "limit": 50,
                "offset": 0,
                "has_more": False,
            }
        }
    )

    items: list[ModelVersionResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
