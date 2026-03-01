from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import desc, select

from api.db.enums import AgentType, GameStatus, QueueType
from api.db.models import BotProfile, Game, GameMove, User


class MatchesRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_game(self, game: Game) -> Game:
        self.session.add(game)
        await self.session.commit()
        await self.session.refresh(game)
        return game

    async def get_game(self, game_id: UUID) -> Game | None:
        return await self.session.get(Game, game_id)

    async def get_user(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def get_bot_profile(self, user_id: UUID) -> BotProfile | None:
        stmt = select(BotProfile).where(BotProfile.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def save_game(self, game: Game) -> Game:
        self.session.add(game)
        await self.session.commit()
        await self.session.refresh(game)
        return game

    async def create_move(self, move: GameMove) -> GameMove:
        self.session.add(move)
        await self.session.commit()
        await self.session.refresh(move)
        return move

    async def next_ply(self, game_id: UUID) -> int:
        stmt = select(GameMove).where(GameMove.game_id == game_id)
        result = await self.session.execute(stmt)
        return len(list(result.scalars().all()))

    async def get_last_move(self, game_id: UUID) -> GameMove | None:
        stmt = (
            select(GameMove)
            .where(GameMove.game_id == game_id)
            .order_by(desc(GameMove.ply))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def count_incoming_invitations(self, user_id: UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(Game)
            .where(
                Game.player2_id == user_id,
                Game.status == GameStatus.PENDING,
                Game.queue_type == QueueType.CUSTOM,
                Game.player1_agent == AgentType.HUMAN,
                Game.player2_agent == AgentType.HUMAN,
            )
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_incoming_invitations(
        self,
        *,
        user_id: UUID,
        limit: int,
        offset: int,
    ) -> list[Game]:
        stmt = (
            select(Game)
            .where(
                Game.player2_id == user_id,
                Game.status == GameStatus.PENDING,
                Game.queue_type == QueueType.CUSTOM,
                Game.player1_agent == AgentType.HUMAN,
                Game.player2_agent == AgentType.HUMAN,
            )
            .order_by(desc(Game.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
