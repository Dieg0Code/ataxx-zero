from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.db.enums import BotKind


class UserCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "bot-alpha",
                "email": "bot-alpha@example.com",
                "password_hash": None,
                "avatar_url": None,
                "country_code": "CL",
                "is_active": True,
                "is_admin": False,
                "is_bot": True,
                "bot_kind": "model",
                "is_hidden_bot": False,
                "model_version_id": None,
            }
        }
    )

    username: str = Field(min_length=3, max_length=32)
    email: str | None = Field(default=None, max_length=255)
    password_hash: str | None = Field(default=None, max_length=255)
    avatar_url: str | None = Field(default=None, max_length=500)
    country_code: str | None = Field(default=None, min_length=2, max_length=2)

    is_active: bool = True
    is_admin: bool = False

    is_bot: bool = False
    bot_kind: BotKind | None = None
    is_hidden_bot: bool = False
    model_version_id: UUID | None = None


class UserResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "5f6e8d34-292d-434f-a8ff-f48f4f3040f9",
                "username": "bot-alpha",
                "email": "bot-alpha@example.com",
                "avatar_url": None,
                "country_code": "CL",
                "is_active": True,
                "is_admin": False,
                "is_bot": True,
                "bot_kind": "model",
                "is_hidden_bot": False,
                "model_version_id": None,
                "created_at": "2026-02-22T20:20:10.000000+00:00",
                "updated_at": "2026-02-22T20:20:10.000000+00:00",
                "last_seen_at": None,
            }
        },
    )

    id: UUID
    username: str
    email: str | None
    avatar_url: str | None
    country_code: str | None
    is_active: bool
    is_admin: bool
    is_bot: bool
    bot_kind: BotKind | None
    is_hidden_bot: bool
    model_version_id: UUID | None
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime | None


class UserListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 0,
                "limit": 50,
                "offset": 0,
                "has_more": False,
            }
        }
    )

    items: list[UserResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
