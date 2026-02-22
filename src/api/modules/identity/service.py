from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from api.db.models import User
from api.modules.identity.repository import UserRepository
from api.modules.identity.schemas import UserCreateRequest


class IdentityService:
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository

    async def create_user(self, payload: UserCreateRequest) -> User:
        now = datetime.now(timezone.utc)
        user = User(
            username=payload.username,
            email=payload.email,
            password_hash=payload.password_hash,
            avatar_url=payload.avatar_url,
            country_code=payload.country_code,
            is_active=payload.is_active,
            is_admin=payload.is_admin,
            is_bot=payload.is_bot,
            bot_kind=payload.bot_kind,
            is_hidden_bot=payload.is_hidden_bot,
            model_version_id=payload.model_version_id,
            created_at=now,
            updated_at=now,
        )
        try:
            return await self.user_repository.create(user)
        except IntegrityError as exc:
            raise ValueError("Username or email already exists.") from exc

    async def get_user(self, user_id: UUID) -> User | None:
        return await self.user_repository.get_by_id(user_id)

    async def list_users(self, *, limit: int = 50, offset: int = 0) -> tuple[int, list[User]]:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        total = await self.user_repository.count_users()
        users = await self.user_repository.list_recent(limit=safe_limit, offset=safe_offset)
        return total, users
