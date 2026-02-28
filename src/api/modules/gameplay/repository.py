from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, or_, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from api.db.enums import GameStatus
from api.db.models import Game, GameMove, QueueEntry, RatingEvent, TrainingSample, User


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

    async def count_all_games_by_status(self, *, statuses: list[GameStatus]) -> int:
        stmt = select(func.count()).select_from(Game).where(col(Game.status).in_(statuses))
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_recent(self, *, limit: int = 20, offset: int = 0) -> list[Game]:
        stmt = select(Game).order_by(col(Game.created_at).desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent_by_status(
        self,
        *,
        statuses: list[GameStatus],
        limit: int = 20,
        offset: int = 0,
    ) -> list[Game]:
        stmt = (
            select(Game)
            .where(col(Game.status).in_(statuses))
            .order_by(col(Game.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent_ids(self, *, limit: int = 20, offset: int = 0) -> list[UUID]:
        stmt = select(col(Game.id)).order_by(col(Game.created_at).desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return [game_id for game_id in result.scalars().all() if game_id is not None]

    async def list_recent_ids_by_status(
        self,
        *,
        statuses: list[GameStatus],
        limit: int = 20,
        offset: int = 0,
    ) -> list[UUID]:
        stmt = (
            select(col(Game.id))
            .where(col(Game.status).in_(statuses))
            .order_by(col(Game.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [game_id for game_id in result.scalars().all() if game_id is not None]

    async def count_games_for_user(self, user_id: UUID) -> int:
        stmt = (
            select(func.count(func.distinct(col(Game.id))))
            .select_from(Game)
            .where(
                or_(
                    col(Game.player1_id) == user_id,
                    col(Game.player2_id) == user_id,
                    col(Game.created_by_user_id) == user_id,
                )
            )
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def count_games_for_user_by_status(self, *, user_id: UUID, statuses: list[GameStatus]) -> int:
        stmt = (
            select(func.count(func.distinct(col(Game.id))))
            .select_from(Game)
            .where(
                or_(
                    col(Game.player1_id) == user_id,
                    col(Game.player2_id) == user_id,
                    col(Game.created_by_user_id) == user_id,
                )
            )
            .where(col(Game.status).in_(statuses))
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def list_recent_for_user(
        self,
        *,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Game]:
        stmt = (
            select(Game)
            .where(
                or_(
                    col(Game.player1_id) == user_id,
                    col(Game.player2_id) == user_id,
                    col(Game.created_by_user_id) == user_id,
                )
            )
            .order_by(col(Game.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent_for_user_by_status(
        self,
        *,
        user_id: UUID,
        statuses: list[GameStatus],
        limit: int = 20,
        offset: int = 0,
    ) -> list[Game]:
        stmt = (
            select(Game)
            .where(
                or_(
                    col(Game.player1_id) == user_id,
                    col(Game.player2_id) == user_id,
                    col(Game.created_by_user_id) == user_id,
                )
            )
            .where(col(Game.status).in_(statuses))
            .order_by(col(Game.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent_ids_for_user(
        self,
        *,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[UUID]:
        stmt = (
            select(col(Game.id))
            .where(
                or_(
                    col(Game.player1_id) == user_id,
                    col(Game.player2_id) == user_id,
                    col(Game.created_by_user_id) == user_id,
                )
            )
            .order_by(col(Game.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [game_id for game_id in result.scalars().all() if game_id is not None]

    async def list_recent_ids_for_user_by_status(
        self,
        *,
        user_id: UUID,
        statuses: list[GameStatus],
        limit: int = 20,
        offset: int = 0,
    ) -> list[UUID]:
        stmt = (
            select(col(Game.id))
            .where(
                or_(
                    col(Game.player1_id) == user_id,
                    col(Game.player2_id) == user_id,
                    col(Game.created_by_user_id) == user_id,
                )
            )
            .where(col(Game.status).in_(statuses))
            .order_by(col(Game.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [game_id for game_id in result.scalars().all() if game_id is not None]

    async def get_user_by_id(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

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

    async def delete_game(self, game_id: UUID) -> bool:
        exists_stmt = (
            select(func.count())
            .select_from(Game)
            .where(col(Game.id) == game_id)
        )
        exists_result = await self.session.execute(exists_stmt)
        if int(exists_result.scalar_one()) == 0:
            return False

        # Release queue references to this game first.
        await self.session.execute(
            update(QueueEntry)
            .where(col(QueueEntry.matched_game_id) == game_id)
            .values(matched_game_id=None)
        )
        # Remove all training/rating artifacts tied to this game.
        await self.session.execute(delete(TrainingSample).where(col(TrainingSample.game_id) == game_id))
        await self.session.execute(delete(RatingEvent).where(col(RatingEvent.game_id) == game_id))
        # Then remove move stream and game row.
        await self.session.execute(delete(GameMove).where(col(GameMove.game_id) == game_id))
        await self.session.execute(delete(Game).where(col(Game.id) == game_id))
        await self.session.commit()
        return True

    async def force_delete_game(self, game_id: UUID) -> bool:
        exists_result = await self.session.execute(
            text('SELECT 1 FROM "game" WHERE id = :game_id LIMIT 1'),
            {"game_id": str(game_id)},
        )
        if exists_result.first() is None:
            return False

        await self.session.execute(
            text(
                "UPDATE queueentry SET matched_game_id = NULL WHERE matched_game_id = :game_id"
            ),
            {"game_id": str(game_id)},
        )
        await self.session.execute(
            text("DELETE FROM trainingsample WHERE game_id = :game_id"),
            {"game_id": str(game_id)},
        )
        await self.session.execute(
            text("DELETE FROM ratingevent WHERE game_id = :game_id"),
            {"game_id": str(game_id)},
        )
        await self.session.execute(
            text("DELETE FROM gamemove WHERE game_id = :game_id"),
            {"game_id": str(game_id)},
        )
        await self.session.execute(
            text('DELETE FROM "game" WHERE id = :game_id'),
            {"game_id": str(game_id)},
        )
        await self.session.commit()
        return True

    async def get_game_access_ids(
        self,
        game_id: UUID,
    ) -> tuple[UUID | None, UUID | None, UUID | None] | None:
        stmt = text(
            """
            SELECT player1_id, player2_id, created_by_user_id
            FROM game
            WHERE id = :game_id
            """
        )
        result = await self.session.execute(stmt, {"game_id": str(game_id)})
        row = result.first()
        if row is None:
            return None

        def _to_uuid(value: object) -> UUID | None:
            if value is None:
                return None
            try:
                return UUID(str(value))
            except (TypeError, ValueError):
                return None

        return (
            _to_uuid(row[0]),
            _to_uuid(row[1]),
            _to_uuid(row[2]),
        )

    async def get_username_by_id(self, user_id: UUID | None) -> str | None:
        if user_id is None:
            return None
        normalized_id = user_id
        if not isinstance(normalized_id, UUID):
            try:
                normalized_id = UUID(str(normalized_id))
            except (TypeError, ValueError):
                return None
        try:
            user = await self.session.get(User, normalized_id)
        except (TypeError, ValueError):
            return None
        if user is None:
            return None
        return user.username
