from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db.enums import AgentType, GameSource, GameStatus, QueueType
from api.db.models import Game, User
from api.deps.auth import get_current_user_dep
from api.deps.gameplay import get_gameplay_service_dep
from api.modules.gameplay.schemas import GameCreateRequest


class _StubGameplayService:
    def __init__(self) -> None:
        self._games: dict[UUID, Game] = {}

    async def create_game(self, payload: GameCreateRequest, actor_user: User) -> Game:
        game = Game(
            queue_type=QueueType.CASUAL,
            status=GameStatus.PENDING,
            rated=False,
            player1_id=actor_user.id,
            player1_agent=AgentType.HUMAN,
            player2_agent=AgentType.HUMAN,
            source=GameSource.HUMAN,
            is_training_eligible=False,
        )
        game.queue_type = payload.queue_type
        game.player1_agent = payload.player1_agent
        game.player2_agent = payload.player2_agent
        self._games[game.id] = game
        return game

    async def get_game(self, game_id: UUID) -> Game | None:
        return self._games.get(game_id)

    async def list_games(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        actor_user: User | None = None,
    ) -> tuple[int, list[Game]]:
        del limit
        del offset
        del actor_user
        rows = list(self._games.values())
        return len(rows), rows

    async def ensure_can_view_game(self, game_id: UUID, actor_user: User) -> Game:
        del actor_user
        game = self._games.get(game_id)
        if game is None:
            raise LookupError(f"Game not found: {game_id}")
        return game


class TestApiGames(unittest.TestCase):
    def _client_with_stub(self) -> tuple[TestClient, _StubGameplayService]:
        app = create_app()
        stub = _StubGameplayService()
        app.dependency_overrides[get_gameplay_service_dep] = lambda: stub
        app.dependency_overrides[get_current_user_dep] = lambda: User(
            id=uuid4(),
            username="test-user",
            email="test-user@example.com",
            password_hash=None,
            is_active=True,
            is_admin=False,
            is_bot=False,
        )
        return TestClient(app), stub

    def test_create_game(self) -> None:
        client, _ = self._client_with_stub()
        response = client.post(
            "/api/v1/gameplay/games",
            json={
                "queue_type": "casual",
                "player1_agent": "human",
                "player2_agent": "heuristic",
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["queue_type"], "casual")
        self.assertEqual(payload["player2_agent"], "heuristic")
        self.assertIn("id", payload)

    def test_get_game_by_id(self) -> None:
        client, _ = self._client_with_stub()
        created = client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "ranked"},
        )
        game_id = created.json()["id"]

        fetched = client.get(f"/api/v1/gameplay/games/{game_id}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["id"], game_id)

    def test_get_game_not_found(self) -> None:
        client, _ = self._client_with_stub()
        missing_id = uuid4()
        response = client.get(f"/api/v1/gameplay/games/{missing_id}")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Game not found", response.json()["detail"])

    def test_list_games(self) -> None:
        client, _ = self._client_with_stub()
        client.post("/api/v1/gameplay/games", json={})
        client.post("/api/v1/gameplay/games", json={"queue_type": "vs_ai"})

        response = client.get("/api/v1/gameplay/games?limit=10")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["offset"], 0)
        self.assertEqual(payload["limit"], 10)
        self.assertEqual(len(payload["items"]), 2)
        self.assertEqual(payload["items"][1]["queue_type"], "vs_ai")


if __name__ == "__main__":
    unittest.main()
