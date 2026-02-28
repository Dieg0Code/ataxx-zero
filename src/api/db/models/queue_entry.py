from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from api.db.enums import QueueEntryStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class QueueEntry(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("season_id", "user_id", name="uq_queueentry_season_user"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    season_id: UUID = Field(foreign_key="season.id", index=True, nullable=False)
    user_id: UUID = Field(foreign_key="user.id", index=True, nullable=False)
    rating_snapshot: float = Field(nullable=False)
    status: QueueEntryStatus = Field(default=QueueEntryStatus.WAITING, index=True)
    matched_game_id: UUID | None = Field(default=None, foreign_key="game.id", index=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)
    matched_at: datetime | None = Field(default=None)
    canceled_at: datetime | None = Field(default=None)
