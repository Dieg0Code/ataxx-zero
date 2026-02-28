from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class QueueJoinResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queue_id": "5f6e8d34-292d-434f-a8ff-f48f4f3040f9",
                "status": "matched",
                "season_id": "4aaddf8e-ca81-4347-a278-f6f7be86c6d0",
                "game_id": "44efed45-d197-4416-bc45-d1cc804f3936",
                "matched_with": "bot",
                "created_at": "2026-02-23T01:10:00",
                "updated_at": "2026-02-23T01:10:02",
            }
        }
    )

    queue_id: UUID
    status: Literal["waiting", "matched", "canceled"]
    season_id: UUID
    game_id: UUID | None
    matched_with: Literal["human", "bot"] | None = None
    created_at: datetime
    updated_at: datetime


class QueueStatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queue_id": "5f6e8d34-292d-434f-a8ff-f48f4f3040f9",
                "status": "waiting",
                "season_id": "4aaddf8e-ca81-4347-a278-f6f7be86c6d0",
                "game_id": None,
                "matched_with": None,
                "created_at": "2026-02-23T01:10:00",
                "updated_at": "2026-02-23T01:10:00",
            }
        }
    )

    queue_id: UUID | None
    status: Literal["idle", "waiting", "matched", "canceled"]
    season_id: UUID | None
    game_id: UUID | None
    matched_with: Literal["human", "bot"] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class QueueLeaveResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "left_queue": True,
                "status": "canceled",
            }
        }
    )

    left_queue: bool
    status: Literal["idle", "canceled"]


class QueueDecisionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision": "accepted",
                "queue_id": "5f6e8d34-292d-434f-a8ff-f48f4f3040f9",
                "status": "matched",
                "game_id": "44efed45-d197-4416-bc45-d1cc804f3936",
                "updated_at": "2026-02-23T01:10:05",
            }
        }
    )

    decision: Literal["accepted", "rejected"]
    queue_id: UUID
    status: Literal["waiting", "matched", "canceled"]
    game_id: UUID | None
    updated_at: datetime
