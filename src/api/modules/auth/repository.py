from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.db.models import AuthRefreshToken, User


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_user(self, user: User) -> User:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def get_user_by_id(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def get_user_by_username_or_email(self, value: str) -> User | None:
        by_username = await self.session.execute(
            select(User).where(User.username == value)
        )
        user = by_username.scalars().first()
        if user is not None:
            return user
        by_email = await self.session.execute(select(User).where(User.email == value))
        return by_email.scalars().first()

    async def create_refresh_token(self, token: AuthRefreshToken) -> AuthRefreshToken:
        self.session.add(token)
        await self.session.commit()
        await self.session.refresh(token)
        return token

    async def get_refresh_token(self, token_hash: str) -> AuthRefreshToken | None:
        stmt = select(AuthRefreshToken).where(AuthRefreshToken.token_hash == token_hash)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def revoke_refresh_token(
        self, token: AuthRefreshToken, revoked_at: datetime
    ) -> AuthRefreshToken:
        token.revoked_at = revoked_at
        self.session.add(token)
        await self.session.commit()
        await self.session.refresh(token)
        return token
