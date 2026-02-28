from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RatingEvent(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    game_id: UUID = Field(foreign_key="game.id", index=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    season_id: UUID = Field(foreign_key="season.id", index=True)
    rating_before: float
    rating_after: float
    delta: float
    before_league: str | None = None
    before_division: str | None = None
    before_lp: int | None = None
    after_league: str | None = None
    after_division: str | None = None
    after_lp: int | None = None
    transition_type: str = "stable"
    major_promo_name: str | None = None
    created_at: datetime = Field(default_factory=utcnow, nullable=False)




