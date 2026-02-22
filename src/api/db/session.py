from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

from api.config.settings import Settings, get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    settings: Settings = get_settings()
    return create_async_engine(
        settings.sqlalchemy_database_url,
        echo=settings.db_echo,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_s,
        pool_recycle=settings.db_pool_recycle_s,
        connect_args={
            "server_settings": {
                "timezone": settings.db_timezone,
            }
        },
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def init_db() -> None:
    """Create mapped tables for local/dev bootstrap."""
    from api.db import models as _models

    del _models
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a transactional async session."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
