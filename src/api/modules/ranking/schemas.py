from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SeasonCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Season 1",
                "starts_at": "2026-03-01T00:00:00+00:00",
                "ends_at": None,
                "is_active": True,
            }
        }
    )

    name: str = Field(min_length=1, max_length=120)
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    is_active: bool = True


class SeasonResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "4aaddf8e-ca81-4347-a278-f6f7be86c6d0",
                "name": "Season 1",
                "starts_at": "2026-03-01T00:00:00+00:00",
                "ends_at": None,
                "is_active": True,
            }
        },
    )

    id: UUID
    name: str
    starts_at: datetime
    ends_at: datetime | None
    is_active: bool


class RatingResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "5f9db46d-47b6-4d1f-97e3-fffd3c4a4226",
                "user_id": "fa7bf995-fd2d-47f1-b76c-d405eb7ca8c5",
                "season_id": "4aaddf8e-ca81-4347-a278-f6f7be86c6d0",
                "rating": 1215.7,
                "games_played": 12,
                "wins": 7,
                "losses": 4,
                "draws": 1,
                "updated_at": "2026-03-20T12:40:00+00:00",
            }
        },
    )

    id: UUID
    user_id: UUID
    season_id: UUID
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    updated_at: datetime


class LeaderboardEntryResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "season_id": "4aaddf8e-ca81-4347-a278-f6f7be86c6d0",
                "user_id": "fa7bf995-fd2d-47f1-b76c-d405eb7ca8c5",
                "rank": 1,
                "rating": 1215.7,
                "wins": 7,
                "losses": 4,
                "draws": 1,
                "win_rate": 0.58,
                "computed_at": "2026-03-20T12:45:00+00:00",
            }
        },
    )

    season_id: UUID
    user_id: UUID
    rank: int
    rating: float
    wins: int
    losses: int
    draws: int
    win_rate: float
    computed_at: datetime


class LeaderboardListResponse(BaseModel):
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

    items: list[LeaderboardEntryResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
