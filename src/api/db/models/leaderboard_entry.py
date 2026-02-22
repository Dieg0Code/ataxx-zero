from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LeaderboardEntry(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("season_id", "user_id", name="uq_leaderboard_season_user"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    season_id: UUID = Field(foreign_key="season.id", index=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    rank: int = Field(ge=1, index=True)
    rating: float = Field(index=True)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    draws: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0)
    computed_at: datetime = Field(default_factory=utcnow, nullable=False)
