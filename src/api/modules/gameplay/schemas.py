from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.db.enums import (
    AgentType,
    GameSource,
    GameStatus,
    PlayerSide,
    QueueType,
    TerminationReason,
    WinnerSide,
)


class BoardStatePayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        }
    )

    grid: list[list[int]]
    current_player: int
    half_moves: int


class MoveRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "mode": "fast",
            }
        }
    )

    board: BoardStatePayload
    mode: Literal[
        "fast",
        "strong",
        "heuristic_easy",
        "heuristic_normal",
        "heuristic_hard",
        "random",
    ] = "fast"


class MovePayload(BaseModel):
    r1: int = Field(ge=0)
    c1: int = Field(ge=0)
    r2: int = Field(ge=0)
    c2: int = Field(ge=0)


class MoveResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
                "action_idx": 42,
                "value": 0.27,
                "mode": "fast",
            }
        }
    )

    move: MovePayload | None
    action_idx: int
    value: float
    mode: Literal[
        "fast",
        "strong",
        "heuristic_easy",
        "heuristic_normal",
        "heuristic_hard",
        "random",
    ]


class ManualMoveRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
            }
        }
    )

    board: BoardStatePayload
    move: MovePayload
    mode: Literal[
        "manual",
        "fast",
        "strong",
        "heuristic_easy",
        "heuristic_normal",
        "heuristic_hard",
        "random",
    ] = "manual"


class GameCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queue_type": "ranked",
                "rated": True,
                "player2_id": "c7fcff51-1564-4eb4-8b4b-157dff7dc132",
                "player1_agent": "human",
                "player2_agent": "human",
                "source": "human",
                "is_training_eligible": False,
            }
        }
    )

    season_id: UUID | None = None
    queue_type: QueueType = QueueType.CASUAL
    status: GameStatus = GameStatus.PENDING
    rated: bool = False

    player1_id: UUID | None = None
    player2_id: UUID | None = None
    player1_agent: AgentType = AgentType.HUMAN
    player2_agent: AgentType = AgentType.HUMAN
    model_version_id: UUID | None = None

    source: GameSource = GameSource.HUMAN
    quality_score: float | None = None
    is_training_eligible: bool = False


class GameResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "44efed45-d197-4416-bc45-d1cc804f3936",
                "season_id": None,
                "queue_type": "casual",
                "status": "in_progress",
                "rated": False,
                "player1_id": "e3886f90-59e4-4ef6-86ae-d0f5f1dcaf33",
                "player2_id": "c7fcff51-1564-4eb4-8b4b-157dff7dc132",
                "created_by_user_id": "e3886f90-59e4-4ef6-86ae-d0f5f1dcaf33",
                "player1_agent": "human",
                "player2_agent": "human",
                "model_version_id": None,
                "winner_side": None,
                "winner_user_id": None,
                "termination_reason": None,
                "source": "human",
                "quality_score": None,
                "is_training_eligible": False,
            }
        },
    )

    id: UUID
    season_id: UUID | None
    queue_type: QueueType
    status: GameStatus
    rated: bool
    player1_id: UUID | None
    player2_id: UUID | None
    created_by_user_id: UUID | None = None
    player1_username: str | None = None
    player2_username: str | None = None
    player1_agent: AgentType
    player2_agent: AgentType
    model_version_id: UUID | None
    winner_side: WinnerSide | None
    winner_user_id: UUID | None
    termination_reason: TerminationReason | None
    source: GameSource
    quality_score: float | None
    is_training_eligible: bool


class StoredMoveResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "1f90ad99-2c2f-428d-b807-c302c44c1728",
                "game_id": "44efed45-d197-4416-bc45-d1cc804f3936",
                "ply": 0,
                "player_side": "p1",
                "r1": 0,
                "c1": 0,
                "r2": 1,
                "c2": 1,
                "board_before": {
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
                "board_after": {
                    "grid": [
                        [1, 0, 0, 0, 0, 0, -1],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 0, 0, 0, 0, 1],
                    ],
                    "current_player": -1,
                    "half_moves": 1,
                },
                "mode": "fast",
                "action_idx": 42,
                "value": 0.12,
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
    board_before: dict[str, object] | None
    board_after: dict[str, object] | None
    mode: str
    action_idx: int
    value: float


class GameReplayResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "game": {
                    "id": "44efed45-d197-4416-bc45-d1cc804f3936",
                    "season_id": None,
                    "queue_type": "casual",
                    "status": "in_progress",
                    "rated": False,
                    "player1_id": "e3886f90-59e4-4ef6-86ae-d0f5f1dcaf33",
                    "player2_id": "c7fcff51-1564-4eb4-8b4b-157dff7dc132",
                    "player1_agent": "human",
                    "player2_agent": "human",
                    "model_version_id": None,
                    "winner_side": None,
                    "winner_user_id": None,
                    "termination_reason": None,
                    "source": "human",
                    "quality_score": None,
                    "is_training_eligible": False,
                },
                "moves": [
                    {
                        "id": "1f90ad99-2c2f-428d-b807-c302c44c1728",
                        "game_id": "44efed45-d197-4416-bc45-d1cc804f3936",
                        "ply": 0,
                        "player_side": "p1",
                        "r1": 0,
                        "c1": 0,
                        "r2": 1,
                        "c2": 1,
                        "board_before": {
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
                        "board_after": {
                            "grid": [
                                [1, 0, 0, 0, 0, 0, -1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [-1, 0, 0, 0, 0, 0, 1],
                            ],
                            "current_player": -1,
                            "half_moves": 1,
                        },
                        "mode": "fast",
                        "action_idx": 42,
                        "value": 0.12,
                    }
                ],
            }
        }
    )

    game: GameResponse
    moves: list[StoredMoveResponse]


class GameListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 0,
                "limit": 20,
                "offset": 0,
                "has_more": False,
            }
        }
    )

    items: list[GameResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
