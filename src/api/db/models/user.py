from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from api.db.enums import BotKind


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("email", name="uq_users_email"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    username: str = Field(index=True, min_length=3, max_length=32)
    email: str | None = Field(default=None, index=True, max_length=255)
    password_hash: str | None = Field(default=None, max_length=255)
    avatar_url: str | None = Field(default=None, max_length=500)
    country_code: str | None = Field(default=None, min_length=2, max_length=2)

    is_active: bool = Field(default=True, index=True)
    is_admin: bool = Field(default=False, index=True)

    is_bot: bool = Field(default=False, index=True)
    bot_kind: BotKind | None = Field(default=None, index=True)
    is_hidden_bot: bool = Field(default=False, index=True)
    model_version_id: UUID | None = Field(
        default=None,
        foreign_key="modelversion.id",
        index=True,
    )

    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)
    last_seen_at: datetime | None = Field(default=None)




