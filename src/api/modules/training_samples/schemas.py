from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.db.enums import GameSource, PlayerSide, SampleSplit


class TrainingSampleCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "game_id": "1c87be62-217d-4ee2-8f15-85080f43e2e9",
                "move_id": None,
                "ply": 0,
                "player_side": "p1",
                "observation": {
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
                "policy_target": {"42": 1.0},
                "value_target": 1.0,
                "sample_weight": 1.0,
                "split": "train",
                "source": "self_play",
            }
        }
    )

    game_id: UUID
    move_id: UUID | None = None
    ply: int = Field(ge=0)
    player_side: PlayerSide
    observation: dict[str, object]
    policy_target: dict[str, float]
    value_target: float
    sample_weight: float = Field(default=1.0, gt=0.0)
    split: SampleSplit = SampleSplit.TRAIN
    source: GameSource = GameSource.SELF_PLAY


class TrainingSampleResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "726f274e-a7ec-4b67-afad-7cfe9eb07096",
                "game_id": "1c87be62-217d-4ee2-8f15-85080f43e2e9",
                "move_id": None,
                "ply": 0,
                "player_side": "p1",
                "observation": {
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
                "policy_target": {"42": 1.0},
                "value_target": 1.0,
                "sample_weight": 1.0,
                "split": "train",
                "source": "self_play",
                "created_at": "2026-02-22T21:10:00+00:00",
            }
        },
    )

    id: UUID
    game_id: UUID
    move_id: UUID | None
    ply: int
    player_side: PlayerSide
    observation: dict[str, object]
    policy_target: dict[str, float]
    value_target: float
    sample_weight: float
    split: SampleSplit
    source: GameSource
    created_at: datetime


class IngestGameSamplesResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "game_id": "1c87be62-217d-4ee2-8f15-85080f43e2e9",
                "created_count": 68,
                "split": "train",
                "source": "self_play",
            }
        }
    )

    game_id: UUID
    created_count: int
    split: SampleSplit
    source: GameSource


class TrainingSamplesStatsResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 15000,
                "by_split": {"train": 13000, "val": 1500, "test": 500},
                "by_source": {"self_play": 12000, "human": 3000},
            }
        }
    )

    total: int
    by_split: dict[SampleSplit, int]
    by_source: dict[GameSource, int]


class TrainingSampleListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 0,
                "limit": 100,
                "offset": 0,
                "has_more": False,
            }
        }
    )

    items: list[TrainingSampleResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
