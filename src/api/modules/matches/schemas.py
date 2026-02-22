from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.db.enums import AgentType, GameSource, GameStatus, PlayerSide, QueueType


class MatchCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queue_type": "casual",
                "rated": False,
                "player2_id": "f328453e-612b-4dfd-b28a-bd8dd8700dc1",
                "player1_agent": "human",
                "player2_agent": "human",
                "source": "human",
                "is_training_eligible": False,
            }
        }
    )

    season_id: UUID | None = None
    queue_type: QueueType = QueueType.CASUAL
    rated: bool = False
    player1_id: UUID | None = None
    player2_id: UUID | None = None
    player1_agent: AgentType = AgentType.HUMAN
    player2_agent: AgentType = AgentType.HUMAN
    model_version_id: UUID | None = None
    source: GameSource = GameSource.HUMAN
    is_training_eligible: bool = False


class MatchMovePayload(BaseModel):
    r1: int = Field(ge=0)
    c1: int = Field(ge=0)
    r2: int = Field(ge=0)
    c2: int = Field(ge=0)


class MatchMoveRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1}, "pass_turn": False}
        }
    )

    move: MatchMovePayload | None = None
    pass_turn: bool = False


class MatchMoveResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "8cdca658-1257-44b8-b338-a568ecad53c3",
                "game_id": "55127c32-ec70-4ef3-8b6e-f28588f16cb0",
                "ply": 0,
                "player_side": "p1",
                "r1": 0,
                "c1": 0,
                "r2": 1,
                "c2": 1,
                "action_idx": 42,
                "mode": "authoritative",
            }
        },
    )

    id: UUID
    game_id: UUID
    ply: int
    player_side: PlayerSide
    r1: int | None
    c1: int | None
    r2: int | None
    c2: int | None
    action_idx: int
    mode: str


class MatchStateResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "game_id": "55127c32-ec70-4ef3-8b6e-f28588f16cb0",
                "status": "in_progress",
                "board": {
                    "grid": [
                        [1, 0, 0, 0, 0, 0, -1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 0, 0, 0, 0, 1],
                    ],
                    "current_player": 1,
                    "half_moves": 0,
                },
                "next_player_side": "p1",
                "legal_moves": [{"r1": 0, "c1": 0, "r2": 1, "c2": 1}],
            }
        }
    )

    game_id: UUID
    status: GameStatus
    board: dict[str, object]
    next_player_side: PlayerSide
    legal_moves: list[MatchMovePayload]
