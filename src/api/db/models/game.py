from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel

from api.db.enums import (
    AgentType,
    GameSource,
    GameStatus,
    QueueType,
    TerminationReason,
    WinnerSide,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Game(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)

    season_id: UUID | None = Field(default=None, foreign_key="season.id", index=True)
    queue_type: QueueType = Field(default=QueueType.CASUAL, index=True)
    status: GameStatus = Field(default=GameStatus.PENDING, index=True)
    rated: bool = Field(default=False, index=True)

    player1_id: UUID | None = Field(default=None, foreign_key="user.id", index=True)
    player2_id: UUID | None = Field(default=None, foreign_key="user.id", index=True)
    created_by_user_id: UUID | None = Field(default=None, foreign_key="user.id", index=True)
    player1_agent: AgentType = Field(default=AgentType.HUMAN)
    player2_agent: AgentType = Field(default=AgentType.HUMAN)
    model_version_id: UUID | None = Field(
        default=None,
        foreign_key="modelversion.id",
        index=True,
    )

    winner_side: WinnerSide | None = Field(default=None, index=True)
    winner_user_id: UUID | None = Field(default=None, foreign_key="user.id", index=True)
    termination_reason: TerminationReason | None = Field(default=None, index=True)

    source: GameSource = Field(default=GameSource.HUMAN, index=True)
    quality_score: float | None = Field(default=None)
    is_training_eligible: bool = Field(default=False, index=True)

    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    started_at: datetime | None = Field(default=None)
    ended_at: datetime | None = Field(default=None)




