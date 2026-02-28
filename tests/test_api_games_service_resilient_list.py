from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

from api.db.enums import AgentType, GameSource, GameStatus, QueueType
from api.db.models import Game, User
from api.modules.gameplay.repository import GameRepository
from api.modules.gameplay.service import GameplayService


class _CorruptingRepository:
    def __init__(self) -> None:
        self.good_game_id = uuid4()
        self.bad_game_id = uuid4()

    async def count_games_for_user(self, user_id: UUID) -> int:
        return 2

    async def list_recent_for_user(self, *, user_id: UUID, limit: int, offset: int) -> list[Game]:
        raise ValueError("invalid enum value from legacy row")

    async def list_recent_ids_for_user(self, *, user_id: UUID, limit: int, offset: int) -> list[UUID]:
        return [self.bad_game_id, self.good_game_id]

    async def get_by_id(self, game_id: UUID) -> Game | None:
        if game_id == self.bad_game_id:
            raise ValueError("corrupt row")
        return Game(
            id=self.good_game_id,
            queue_type=QueueType.CASUAL,
            status=GameStatus.PENDING,
            rated=False,
            player1_id=None,
            player2_id=None,
            created_by_user_id=None,
            player1_agent=AgentType.HUMAN,
            player2_agent=AgentType.HUMAN,
            model_version_id=None,
            winner_side=None,
            winner_user_id=None,
            termination_reason=None,
            source=GameSource.HUMAN,
            quality_score=None,
            is_training_eligible=False,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )


class TestGameplayServiceResilientList(unittest.TestCase):
    def test_list_games_skips_corrupt_rows_in_fallback(self) -> None:
        async def _run() -> None:
            raw_repository = _CorruptingRepository()
            repository = cast(GameRepository, raw_repository)
            service = GameplayService(game_repository=repository, ranking_service=None)
            actor_user = User(
                id=uuid4(),
                username="user-a",
                email=None,
                password_hash=None,
                is_active=True,
                is_admin=False,
                is_bot=False,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_seen_at=None,
            )

            total, games = await service.list_games(
                limit=8,
                offset=0,
                actor_user=actor_user,
            )

            self.assertEqual(total, 2)
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0].id, raw_repository.good_game_id)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
