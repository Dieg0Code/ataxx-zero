from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Rating(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", "season_id", name="uq_rating_user_season"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    season_id: UUID = Field(foreign_key="season.id", index=True)
    rating: float = Field(default=1200.0)
    games_played: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)




