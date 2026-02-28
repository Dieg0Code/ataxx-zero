from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Season(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("name", name="uq_season_name"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True, min_length=1, max_length=120)
    starts_at: datetime = Field(default_factory=utcnow, nullable=False)
    ends_at: datetime | None = Field(default=None)
    is_active: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)




