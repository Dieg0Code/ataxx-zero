from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.db.models import LeaderboardEntry, Rating, RatingEvent, Season


class RankingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_season(self, season: Season) -> Season:
        self.session.add(season)
        await self.session.commit()
        await self.session.refresh(season)
        return season

    async def get_active_season(self) -> Season | None:
        stmt = select(Season).where(Season.is_active)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_rating(self, user_id: UUID, season_id: UUID) -> Rating | None:
        stmt = select(Rating).where(
            Rating.user_id == user_id,
            Rating.season_id == season_id,
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create_rating(self, rating: Rating) -> Rating:
        self.session.add(rating)
        await self.session.commit()
        await self.session.refresh(rating)
        return rating

    async def save_rating(self, rating: Rating) -> Rating:
        self.session.add(rating)
        await self.session.commit()
        await self.session.refresh(rating)
        return rating

    async def create_rating_event(self, event: RatingEvent) -> RatingEvent:
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event

    async def list_ratings_for_season(self, season_id: UUID, limit: int = 200) -> list[Rating]:
        stmt = select(Rating).where(Rating.season_id == season_id).limit(limit)
        result = await self.session.execute(stmt)
        ratings = list(result.scalars().all())
        ratings.sort(key=lambda row: row.rating, reverse=True)
        return ratings

    async def replace_leaderboard_entries(
        self,
        season_id: UUID,
        entries: list[LeaderboardEntry],
    ) -> list[LeaderboardEntry]:
        existing = await self.get_leaderboard_entries(season_id=season_id, limit=5000)
        for row in existing:
            await self.session.delete(row)
        if entries:
            self.session.add_all(entries)
        await self.session.commit()
        # Refresh one by one for stable ORM state.
        for entry in entries:
            await self.session.refresh(entry)
        return entries

    async def get_leaderboard_entries(
        self,
        season_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LeaderboardEntry]:
        stmt = select(LeaderboardEntry).where(LeaderboardEntry.season_id == season_id)
        result = await self.session.execute(stmt)
        entries = list(result.scalars().all())
        entries.sort(key=lambda row: row.rank)
        return entries[offset : offset + limit]

    async def count_leaderboard_entries(self, season_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(LeaderboardEntry)
            .where(LeaderboardEntry.season_id == season_id)
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)
