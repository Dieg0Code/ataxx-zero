from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuthRefreshToken(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_auth_refresh_tokens_token_hash"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    token_hash: str = Field(index=True, min_length=64, max_length=128)
    expires_at: datetime = Field(nullable=False, index=True)
    revoked_at: datetime | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
