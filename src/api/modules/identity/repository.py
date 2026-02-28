from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from api.db.enums import AgentType
from api.db.models import BotProfile, User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def count_users(self) -> int:
        stmt = select(func.count()).select_from(User)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_recent(self, *, limit: int = 50, offset: int = 0) -> list[User]:
        stmt = select(User).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_playable_bots(self) -> int:
        stmt = (
            select(func.count())
            .select_from(User)
            .join(BotProfile, col(BotProfile.user_id) == col(User.id))
            .where(col(User.is_bot))
            .where(col(User.is_active))
            .where(col(BotProfile.enabled))
            .where(col(BotProfile.agent_type) != AgentType.HUMAN)
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_playable_bots(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[tuple[User, BotProfile]]:
        stmt = (
            select(User, BotProfile)
            .join(BotProfile, col(BotProfile.user_id) == col(User.id))
            .where(col(User.is_bot))
            .where(col(User.is_active))
            .where(col(BotProfile.enabled))
            .where(col(BotProfile.agent_type) != AgentType.HUMAN)
            .order_by(col(User.username))
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def count_public_players(self, *, query: str | None = None) -> int:
        stmt = (
            select(func.count())
            .select_from(User)
            .where(col(User.is_active))
            .where(~(col(User.is_bot) & col(User.is_hidden_bot)))
        )
        if query:
            stmt = stmt.where(col(User.username).ilike(f"%{query}%"))
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_public_players(
        self, *, limit: int = 50, offset: int = 0, query: str | None = None
    ) -> list[tuple[User, BotProfile | None]]:
        stmt_base = (
            select(User, BotProfile)
            .outerjoin(BotProfile, col(BotProfile.user_id) == col(User.id))
            .where(col(User.is_active))
            .where(~(col(User.is_bot) & col(User.is_hidden_bot)))
        )
        stmt = (
            stmt_base.where(col(User.username).ilike(f"%{query}%"))
            if query
            else stmt_base
        )
        stmt = stmt.order_by(col(User.username)).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]
