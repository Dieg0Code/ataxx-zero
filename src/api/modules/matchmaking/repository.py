from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from api.db.enums import AgentType, QueueEntryStatus, QueueType
from api.db.models import (
    BotProfile,
    Game,
    GameMove,
    QueueEntry,
    RatingEvent,
    TrainingSample,
    User,
)


class MatchmakingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_waiting_entry_for_user(self, *, season_id: UUID, user_id: UUID) -> QueueEntry | None:
        stmt = (
            select(QueueEntry)
            .where(col(QueueEntry.season_id) == season_id)
            .where(col(QueueEntry.user_id) == user_id)
            .where(col(QueueEntry.status) == QueueEntryStatus.WAITING)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_entry_for_user(self, *, season_id: UUID, user_id: UUID) -> QueueEntry | None:
        stmt = (
            select(QueueEntry)
            .where(col(QueueEntry.season_id) == season_id)
            .where(col(QueueEntry.user_id) == user_id)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_matched_entry_for_user(self, *, season_id: UUID, user_id: UUID) -> QueueEntry | None:
        stmt = (
            select(QueueEntry)
            .where(col(QueueEntry.season_id) == season_id)
            .where(col(QueueEntry.user_id) == user_id)
            .where(col(QueueEntry.status) == QueueEntryStatus.MATCHED)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create_entry(self, entry: QueueEntry) -> QueueEntry:
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def save_entry(self, entry: QueueEntry) -> QueueEntry:
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def list_waiting_human_entries(
        self,
        *,
        season_id: UUID,
        user_id: UUID,
        min_rating: float,
        max_rating: float,
        limit: int = 100,
    ) -> list[QueueEntry]:
        stmt = (
            select(QueueEntry)
            .join(User, col(User.id) == col(QueueEntry.user_id))
            .where(col(QueueEntry.season_id) == season_id)
            .where(col(QueueEntry.status) == QueueEntryStatus.WAITING)
            .where(col(QueueEntry.user_id) != user_id)
            .where(col(User.is_bot) == False)  # noqa: E712
            .where(col(QueueEntry.rating_snapshot) >= min_rating)
            .where(col(QueueEntry.rating_snapshot) <= max_rating)
            .order_by(col(QueueEntry.created_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_playable_bots(self, *, include_hidden: bool = False, limit: int = 200) -> list[tuple[User, BotProfile]]:
        stmt = (
            select(User, BotProfile)
            .join(BotProfile, col(BotProfile.user_id) == col(User.id))
            .where(col(User.is_bot))
            .where(col(User.is_active))
            .where(col(BotProfile.enabled))
            .where(col(BotProfile.agent_type) != AgentType.HUMAN)
            .order_by(col(User.username))
            .limit(limit)
        )
        if not include_hidden:
            stmt = stmt.where(col(User.is_hidden_bot) == False)  # noqa: E712
        result = await self.session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def cancel_waiting_entry(
        self,
        *,
        season_id: UUID,
        user_id: UUID,
        now: datetime,
    ) -> bool:
        entry = await self.get_waiting_entry_for_user(season_id=season_id, user_id=user_id)
        if entry is None:
            return False
        entry.status = QueueEntryStatus.CANCELED
        entry.canceled_at = now
        entry.updated_at = now
        await self.save_entry(entry)
        return True

    async def get_entry_by_id(self, queue_id: UUID) -> QueueEntry | None:
        return await self.session.get(QueueEntry, queue_id)

    async def list_matched_entries_for_game(self, *, season_id: UUID, game_id: UUID) -> list[QueueEntry]:
        stmt = (
            select(QueueEntry)
            .where(col(QueueEntry.season_id) == season_id)
            .where(col(QueueEntry.status) == QueueEntryStatus.MATCHED)
            .where(col(QueueEntry.matched_game_id) == game_id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_game_moves(self, *, game_id: UUID) -> int:
        stmt = select(func.count()).select_from(GameMove).where(col(GameMove.game_id) == game_id)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def delete_game_cascade(self, *, game_id: UUID) -> None:
        await self.session.execute(delete(TrainingSample).where(col(TrainingSample.game_id) == game_id))
        await self.session.execute(delete(RatingEvent).where(col(RatingEvent.game_id) == game_id))
        await self.session.execute(delete(GameMove).where(col(GameMove.game_id) == game_id))
        await self.session.execute(delete(Game).where(col(Game.id) == game_id))
        await self.session.commit()

    async def get_last_ranked_bot_opponent_user_id(self, *, season_id: UUID, user_id: UUID) -> UUID | None:
        stmt = (
            select(Game.player2_id)
            .join(User, col(User.id) == col(Game.player2_id))
            .where(col(Game.season_id) == season_id)
            .where(col(Game.queue_type) == QueueType.RANKED)
            .where(col(Game.player1_id) == user_id)
            .where(col(User.is_bot))
            .order_by(col(Game.started_at).desc(), col(Game.created_at).desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
