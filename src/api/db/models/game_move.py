from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from api.db.enums import PlayerSide


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GameMove(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("game_id", "ply", name="uq_game_moves_game_ply"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    game_id: UUID = Field(foreign_key="game.id", index=True)
    ply: int = Field(ge=0, index=True)
    player_side: PlayerSide = Field(index=True)

    r1: int | None = Field(default=None, ge=0)
    c1: int | None = Field(default=None, ge=0)
    r2: int | None = Field(default=None, ge=0)
    c2: int | None = Field(default=None, ge=0)
    board_before: dict[str, object] | None = Field(default=None, sa_type=JSON)
    board_after: dict[str, object] | None = Field(default=None, sa_type=JSON)

    mode: str = Field(default="fast", max_length=16)
    action_idx: int = Field(ge=0)
    value: float
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
