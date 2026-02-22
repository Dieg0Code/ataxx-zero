from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("name", name="uq_model_versions_name"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True, min_length=1, max_length=120)
    hf_repo_id: str | None = Field(default=None, max_length=255)
    hf_revision: str | None = Field(default=None, max_length=255)
    checkpoint_uri: str | None = Field(default=None, max_length=500)
    onnx_uri: str | None = Field(default=None, max_length=500)
    is_active: bool = Field(default=False, index=True)
    notes: str | None = Field(default=None, max_length=2000)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
