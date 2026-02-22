from __future__ import annotations

from uuid import UUID

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.db.models import Game, GameMove


class GameRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, game: Game) -> Game:
        self.session.add(game)
        await self.session.commit()
        await self.session.refresh(game)
        return game

    async def get_by_id(self, game_id: UUID) -> Game | None:
        return await self.session.get(Game, game_id)

    async def count_all_games(self) -> int:
        stmt = select(func.count()).select_from(Game)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_recent(self, *, limit: int = 20, offset: int = 0) -> list[Game]:
        stmt = select(Game).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_games_for_user(self, user_id: UUID) -> int:
        p1_stmt = select(func.count()).select_from(Game).where(Game.player1_id == user_id)
        p2_stmt = select(func.count()).select_from(Game).where(Game.player2_id == user_id)
        p1_result = await self.session.execute(p1_stmt)
        p2_result = await self.session.execute(p2_stmt)
        p1_count = int(p1_result.scalar_one())
        p2_count = int(p2_result.scalar_one())
        # Games where both players are same user should count once.
        both_stmt = (
            select(func.count())
            .select_from(Game)
            .where(Game.player1_id == user_id)
            .where(Game.player2_id == user_id)
        )
        both_result = await self.session.execute(both_stmt)
        both_count = int(both_result.scalar_one())
        return p1_count + p2_count - both_count

    async def list_recent_for_user(
        self,
        *,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Game]:
        p1_stmt = select(Game).where(Game.player1_id == user_id)
        p2_stmt = select(Game).where(Game.player2_id == user_id)
        p1_result = await self.session.execute(p1_stmt)
        p2_result = await self.session.execute(p2_stmt)
        merged: dict[UUID, Game] = {}
        for row in list(p1_result.scalars().all()) + list(p2_result.scalars().all()):
            merged[row.id] = row
        rows = list(merged.values())
        return rows[offset : offset + limit]

    async def create_move(self, move: GameMove) -> GameMove:
        self.session.add(move)
        await self.session.commit()
        await self.session.refresh(move)
        return move

    async def next_ply(self, game_id: UUID) -> int:
        stmt = select(GameMove).where(GameMove.game_id == game_id)
        result = await self.session.execute(stmt)
        moves = list(result.scalars().all())
        return len(moves)

    async def list_moves(self, game_id: UUID, limit: int = 200) -> list[GameMove]:
        stmt = select(GameMove).where(GameMove.game_id == game_id).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def save_game(self, game: Game) -> Game:
        self.session.add(game)
        await self.session.commit()
        await self.session.refresh(game)
        return game
