from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.models import User
from api.db.session import get_session
from api.deps.inference import get_inference_service_dep
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.serialization import board_to_state
from inference.service import InferenceResult

del _models


class _StubInferenceService:
    def predict(self, board: AtaxxBoard, mode: str = "fast") -> InferenceResult:
        if board.is_game_over():
            return InferenceResult(
                move=None,
                action_idx=ACTION_SPACE.pass_index,
                value=1.0,
                mode="fast" if mode == "fast" else "strong",
            )
        move = (0, 0, 1, 1)
        return InferenceResult(
            move=move,
            action_idx=ACTION_SPACE.encode(move),
            value=0.31,
            mode="fast" if mode == "fast" else "strong",
        )


class TestApiGamesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "integration.db"
        cls.database_url = f"sqlite+aiosqlite:///{db_path}"
        cls.engine = create_async_engine(cls.database_url, echo=False)
        cls.sessionmaker = async_sessionmaker(
            bind=cls.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async def _init_db() -> None:
            async with cls.engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)

        asyncio.run(_init_db())

        app = create_app()

        async def _get_session_override() -> AsyncGenerator[AsyncSession, None]:
            async with cls.sessionmaker() as session:
                try:
                    yield session
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_session] = _get_session_override
        app.dependency_overrides[get_inference_service_dep] = _StubInferenceService
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

        async def _dispose() -> None:
            await cls.engine.dispose()

        asyncio.run(_dispose())
        cls.tmpdir.cleanup()

    def test_create_get_and_list_games_with_real_db(self) -> None:
        user = self._register_and_login("gp-u1", "gp-u1@example.com")
        outsider = self._register_and_login("gp-u2", "gp-u2@example.com")

        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "queue_type": "ranked",
                "rated": True,
                "player1_agent": "human",
                "player2_agent": "heuristic",
                "source": "human",
            },
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.json()
        game_id = created["id"]
        self.assertEqual(created["queue_type"], "ranked")
        self.assertTrue(created["rated"])

        get_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(get_resp.status_code, 200)
        fetched = get_resp.json()
        self.assertEqual(fetched["id"], game_id)
        self.assertEqual(fetched["player2_agent"], "heuristic")

        list_resp = self.client.get(
            "/api/v1/gameplay/games?limit=10",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_resp.status_code, 200)
        page = list_resp.json()
        self.assertGreaterEqual(page["total"], 1)
        all_games = page["items"]
        ids = {entry["id"] for entry in all_games}
        self.assertIn(game_id, ids)

        outsider_get = self.client.get(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_get.status_code, 403)

    def test_store_and_list_game_moves(self) -> None:
        user = self._register_and_login("gp-u3", "gp-u3@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        board = AtaxxBoard()
        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(board), "mode": "fast"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)
        stored = move_resp.json()
        self.assertEqual(stored["game_id"], game_id)
        self.assertEqual(stored["r1"], 0)
        self.assertEqual(stored["c1"], 0)
        self.assertEqual(stored["r2"], 1)
        self.assertEqual(stored["c2"], 1)
        self.assertIn("board_before", stored)
        self.assertIn("board_after", stored)
        self.assertEqual(stored["board_before"]["current_player"], 1)
        self.assertEqual(stored["board_after"]["current_player"], -1)

        list_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}/moves",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_resp.status_code, 200)
        moves = list_resp.json()
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["game_id"], game_id)

        replay_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}/replay",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay = replay_resp.json()
        self.assertEqual(replay["game"]["id"], game_id)
        self.assertEqual(len(replay["moves"]), 1)
        self.assertEqual(replay["moves"][0]["game_id"], game_id)

    def test_game_is_auto_finished_on_terminal_board(self) -> None:
        user = self._register_and_login("gp-u4", "gp-u4@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        terminal_board = AtaxxBoard()
        terminal_board.grid[:, :] = 1
        terminal_board.current_player = 1

        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(terminal_board), "mode": "fast"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)

        game_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 200)
        game = game_resp.json()
        self.assertEqual(game["status"], "finished")
        self.assertEqual(game["winner_side"], "p1")
        self.assertEqual(game["termination_reason"], "normal")

    def test_rated_game_updates_ratings(self) -> None:
        admin = self._register_and_login("gp-admin", "gp-admin@example.com")
        self._promote_user_to_admin(admin["user_id"])
        p1 = self._register_and_login("rated-p1", "rated-p1@example.com")
        p2 = self._register_and_login("rated-p2", "rated-p2@example.com")
        p1_id = p1["user_id"]
        p2_id = p2["user_id"]

        season_resp = self.client.post(
            "/api/v1/ranking/seasons",
            json={"name": "S1 Integration", "is_active": True},
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(season_resp.status_code, 201)
        season_id = season_resp.json()["id"]

        game_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "rated": True,
                "season_id": season_id,
                "player1_id": p1_id,
                "player2_id": p2_id,
                "queue_type": "ranked",
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 201)
        game_id = game_resp.json()["id"]

        terminal_board = AtaxxBoard()
        terminal_board.grid[:, :] = 1
        terminal_board.current_player = 1
        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(terminal_board), "mode": "fast"},
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)

        p1_rating_resp = self.client.get(f"/api/v1/ranking/ratings/{p1_id}/{season_id}")
        p2_rating_resp = self.client.get(f"/api/v1/ranking/ratings/{p2_id}/{season_id}")
        self.assertEqual(p1_rating_resp.status_code, 200)
        self.assertEqual(p2_rating_resp.status_code, 200)
        p1_rating = p1_rating_resp.json()
        p2_rating = p2_rating_resp.json()
        self.assertEqual(p1_rating["games_played"], 1)
        self.assertEqual(p2_rating["games_played"], 1)
        self.assertEqual(p1_rating["wins"], 1)
        self.assertEqual(p2_rating["losses"], 1)
        self.assertGreater(p1_rating["rating"], 1200.0)
        self.assertLess(p2_rating["rating"], 1200.0)

        lb_resp = self.client.get(f"/api/v1/ranking/leaderboard/{season_id}?limit=10")
        self.assertEqual(lb_resp.status_code, 200)
        leaderboard = lb_resp.json()["items"]
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard[0]["user_id"], p1_id)
        self.assertEqual(leaderboard[0]["rank"], 1)
        self.assertEqual(leaderboard[1]["user_id"], p2_id)
        self.assertEqual(leaderboard[1]["rank"], 2)

    def _register_and_login(self, username: str, email: str) -> dict[str, str]:
        register = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "supersecret123",
            },
        )
        self.assertEqual(register.status_code, 201)
        user_id = register.json()["id"]

        login = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": username, "password": "supersecret123"},
        )
        self.assertEqual(login.status_code, 200)
        return {"user_id": user_id, "access_token": login.json()["access_token"]}

    def _promote_user_to_admin(self, user_id: str) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                user = await session.get(User, UUID(user_id))
                if user is None:
                    raise AssertionError("User not found for admin promotion")
                user.is_admin = True
                session.add(user)
                await session.commit()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
