from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import desc, select

from api.db.models import Game, GameMove


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
