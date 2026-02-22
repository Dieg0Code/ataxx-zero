from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.db.models import User


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
