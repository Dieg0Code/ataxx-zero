from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AuthRegisterRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "diego",
                "email": "diego@example.com",
                "password": "supersecret123",
            }
        }
    )

    username: str = Field(min_length=3, max_length=32)
    email: str | None = Field(default=None, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class AuthLoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username_or_email": "diego",
                "password": "supersecret123",
            }
        }
    )

    username_or_email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class AuthRefreshRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"refresh_token": "<refresh_jwt>"}}
    )

    refresh_token: str = Field(min_length=16)


class AuthLogoutRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"refresh_token": "<refresh_jwt>"}}
    )

    refresh_token: str = Field(min_length=16)


class AuthUserResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "7fa22272-d5a3-4374-8f89-dfdddb4251f0",
                "username": "diego",
                "email": "diego@example.com",
                "is_active": True,
                "is_admin": False,
                "created_at": "2026-02-22T20:20:10.000000+00:00",
                "updated_at": "2026-02-22T20:20:10.000000+00:00",
            }
        },
    )

    id: UUID
    username: str
    email: str | None
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: datetime


class AuthTokenPairResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "<access_jwt>",
                "refresh_token": "<refresh_jwt>",
                "token_type": "bearer",
            }
        }
    )

    access_token: str
    refresh_token: str
    token_type: str
