from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, not_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from api.db.models import LeaderboardEntry, Rating, RatingEvent, Season, User


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

    async def count_rating_events(self, user_id: UUID, season_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(RatingEvent)
            .where(RatingEvent.user_id == user_id, RatingEvent.season_id == season_id)
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_rating_events(
        self,
        user_id: UUID,
        season_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RatingEvent]:
        stmt = select(RatingEvent).where(
            RatingEvent.user_id == user_id,
            RatingEvent.season_id == season_id,
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        rows.sort(key=lambda row: row.created_at, reverse=True)
        return rows[offset : offset + limit]

    async def get_latest_rating_events_for_user_ids(
        self,
        season_id: UUID,
        user_ids: list[UUID],
    ) -> dict[UUID, RatingEvent]:
        if not user_ids:
            return {}
        stmt = select(RatingEvent).where(
            RatingEvent.season_id == season_id,
            col(RatingEvent.user_id).in_(user_ids),
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        rows.sort(key=lambda row: row.created_at, reverse=True)
        latest: dict[UUID, RatingEvent] = {}
        for row in rows:
            if row.user_id not in latest:
                latest[row.user_id] = row
        return latest

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
        # Ensure DELETE statements are emitted before INSERTs to avoid
        # unique conflicts on (season_id, user_id) during recompute.
        await self.session.flush()
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
        competitor_filter: str = "all",
        username_query: str | None = None,
    ) -> list[LeaderboardEntry]:
        stmt = (
            select(LeaderboardEntry)
            .join(User, col(LeaderboardEntry.user_id) == col(User.id))
            .where(LeaderboardEntry.season_id == season_id)
        )
        if competitor_filter == "bots":
            stmt = stmt.where(col(User.is_bot))
        elif competitor_filter == "humans":
            stmt = stmt.where(not_(col(User.is_bot)))
        if username_query:
            stmt = stmt.where(col(User.username).ilike(f"%{username_query}%"))
        result = await self.session.execute(stmt)
        entries = list(result.scalars().all())
        entries.sort(key=lambda row: row.rank)
        return entries[offset : offset + limit]

    async def count_leaderboard_entries(
        self,
        season_id: UUID,
        competitor_filter: str = "all",
        username_query: str | None = None,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(LeaderboardEntry)
            .join(User, col(LeaderboardEntry.user_id) == col(User.id))
            .where(LeaderboardEntry.season_id == season_id)
        )
        if competitor_filter == "bots":
            stmt = stmt.where(col(User.is_bot))
        elif competitor_filter == "humans":
            stmt = stmt.where(not_(col(User.is_bot)))
        if username_query:
            stmt = stmt.where(col(User.username).ilike(f"%{username_query}%"))
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def get_usernames_for_user_ids(self, user_ids: list[UUID]) -> dict[UUID, str]:
        if not user_ids:
            return {}
        stmt = select(User).where(col(User.id).in_(user_ids))
        result = await self.session.execute(stmt)
        users = list(result.scalars().all())
        return {user.id: user.username for user in users}

    async def get_user_public_info_for_user_ids(
        self,
        user_ids: list[UUID],
    ) -> dict[UUID, tuple[str, bool, str | None]]:
        if not user_ids:
            return {}
        stmt = select(User).where(col(User.id).in_(user_ids))
        result = await self.session.execute(stmt)
        users = list(result.scalars().all())
        return {user.id: (user.username, user.is_bot, user.bot_kind.value if user.bot_kind else None) for user in users}

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)



