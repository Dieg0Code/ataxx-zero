from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RatingEvent(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    game_id: UUID = Field(foreign_key="game.id", index=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    season_id: UUID = Field(foreign_key="season.id", index=True)
    rating_before: float
    rating_after: float
    delta: float
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
