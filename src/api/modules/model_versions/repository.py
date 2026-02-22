from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.db.models import ModelVersion


class ModelVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, version: ModelVersion) -> ModelVersion:
        self.session.add(version)
        await self.session.commit()
        await self.session.refresh(version)
        return version

    async def get_by_id(self, version_id: UUID) -> ModelVersion | None:
        return await self.session.get(ModelVersion, version_id)

    async def count_versions(self) -> int:
        stmt = select(func.count()).select_from(ModelVersion)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_recent(self, *, limit: int = 50, offset: int = 0) -> list[ModelVersion]:
        stmt = select(ModelVersion).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_active(self) -> ModelVersion | None:
        stmt = select(ModelVersion).where(ModelVersion.is_active)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_active(self) -> list[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.is_active)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def save(self, version: ModelVersion) -> ModelVersion:
        self.session.add(version)
        await self.session.commit()
        await self.session.refresh(version)
        return version

    async def deactivate_all_except(self, keep_id: UUID) -> None:
        active_rows = await self.list_active()
        for row in active_rows:
            if row.id != keep_id:
                row.is_active = False
                self.session.add(row)
        await self.session.commit()
