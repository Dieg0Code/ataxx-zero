from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from api.db.enums import GameSource, PlayerSide, SampleSplit


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingSample(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "ply",
            name="uq_training_samples_game_ply",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    game_id: UUID = Field(foreign_key="game.id", index=True)
    move_id: UUID | None = Field(default=None, foreign_key="gamemove.id", index=True)
    ply: int = Field(ge=0, index=True)
    player_side: PlayerSide = Field(index=True)

    observation: dict[str, object] = Field(sa_type=JSON)
    policy_target: dict[str, float] = Field(sa_type=JSON)
    value_target: float
    sample_weight: float = Field(default=1.0, gt=0.0)

    split: SampleSplit = Field(default=SampleSplit.TRAIN, index=True)
    source: GameSource = Field(default=GameSource.SELF_PLAY, index=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
