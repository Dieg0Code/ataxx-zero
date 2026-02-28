from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from api.db.enums import AgentType


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class BotProfile(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", name="uq_bot_profile_user_id"),)

    user_id: UUID = Field(primary_key=True, foreign_key="user.id")
    agent_type: AgentType = Field(index=True)
    heuristic_level: str | None = Field(default=None, max_length=16)
    model_mode: str | None = Field(default=None, max_length=16)
    enabled: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)
