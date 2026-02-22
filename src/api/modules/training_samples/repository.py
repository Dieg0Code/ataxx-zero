from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.db.enums import SampleSplit
from api.db.models import Game, GameMove, TrainingSample


class TrainingSamplesRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, sample: TrainingSample) -> TrainingSample:
        self.session.add(sample)
        await self.session.commit()
        await self.session.refresh(sample)
        return sample

    async def bulk_create(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        if not samples:
            return []
        self.session.add_all(samples)
        await self.session.commit()
        for sample in samples:
            await self.session.refresh(sample)
        return samples

    async def get_by_id(self, sample_id: UUID) -> TrainingSample | None:
        return await self.session.get(TrainingSample, sample_id)

    async def list_samples(
        self,
        *,
        limit: int,
        offset: int = 0,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
    ) -> list[TrainingSample]:
        stmt = select(TrainingSample)
        if split is not None:
            stmt = stmt.where(TrainingSample.split == split)
        if game_id is not None:
            stmt = stmt.where(TrainingSample.game_id == game_id)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_samples(
        self,
        *,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(TrainingSample)
        if split is not None:
            stmt = stmt.where(TrainingSample.split == split)
        if game_id is not None:
            stmt = stmt.where(TrainingSample.game_id == game_id)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_all_samples(
        self,
        *,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
    ) -> list[TrainingSample]:
        stmt = select(TrainingSample)
        if split is not None:
            stmt = stmt.where(TrainingSample.split == split)
        if game_id is not None:
            stmt = stmt.where(TrainingSample.game_id == game_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_game(self, game_id: UUID) -> Game | None:
        return await self.session.get(Game, game_id)

    async def list_game_moves(self, game_id: UUID) -> list[GameMove]:
        stmt = select(GameMove).where(GameMove.game_id == game_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_for_game(self, game_id: UUID) -> int:
        stmt = select(TrainingSample).where(TrainingSample.game_id == game_id)
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        for row in rows:
            await self.session.delete(row)
        await self.session.commit()
        return len(rows)
