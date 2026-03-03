from __future__ import annotations

import asyncio
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.db.enums import AgentType, GameSource, GameStatus, QueueType
from api.db.models import Game
from api.modules.gameplay.router import _to_game_response
from api.modules.gameplay.service import GameplayService


def _sample_game() -> Game:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return Game(
        id=uuid4(),
        queue_type=QueueType.CASUAL,
        status=GameStatus.PENDING,
        rated=False,
        player1_id=uuid4(),
        player2_id=uuid4(),
        created_by_user_id=None,
        player1_agent=AgentType.HUMAN,
        player2_agent=AgentType.HEURISTIC,
        model_version_id=None,
        winner_side=None,
        winner_user_id=None,
        termination_reason=None,
        source=GameSource.HUMAN,
        quality_score=None,
        is_training_eligible=False,
        created_at=now,
        updated_at=now,
    )


class _ServiceWithoutUsernameLookup:
    async def get_player_usernames(self, _game: Game) -> tuple[str | None, str | None]:
        raise AssertionError("Username lookup should not run when include_usernames=False")


class _ServiceWithUsernameLookup:
    async def get_player_usernames(self, _game: Game) -> tuple[str | None, str | None]:
        return "p1-user", "p2-user"


class TestGameplayRouterUnit(unittest.TestCase):
    def test_to_game_response_can_skip_username_lookup(self) -> None:
        async def _run() -> None:
            payload = await _to_game_response(
                gameplay_service=cast(GameplayService, _ServiceWithoutUsernameLookup()),
                game=_sample_game(),
                include_usernames=False,
            )
            self.assertIsNone(payload.player1_username)
            self.assertIsNone(payload.player2_username)

        asyncio.run(_run())

    def test_to_game_response_keeps_username_lookup_when_enabled(self) -> None:
        async def _run() -> None:
            payload = await _to_game_response(
                gameplay_service=cast(GameplayService, _ServiceWithUsernameLookup()),
                game=_sample_game(),
                include_usernames=True,
            )
            self.assertEqual(payload.player1_username, "p1-user")
            self.assertEqual(payload.player2_username, "p2-user")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
