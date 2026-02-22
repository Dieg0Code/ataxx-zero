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

del _models


class TestApiMatchesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "matches_integration.db"
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
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

        async def _dispose() -> None:
            await cls.engine.dispose()

        asyncio.run(_dispose())
        cls.tmpdir.cleanup()

    def test_create_match_and_get_initial_state(self) -> None:
        p1 = self._register_and_login("match-p1", "match-p1@example.com")
        p2 = self._register_and_login("match-p2", "match-p2@example.com")
        outsider = self._register_and_login("match-outside", "match-outside@example.com")

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "queue_type": "casual",
                "player1_id": p1["user_id"],
                "player2_id": p2["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        get_resp = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(get_resp.status_code, 200)

        state_resp = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(state_resp.status_code, 200)
        state = state_resp.json()
        self.assertEqual(state["game_id"], game_id)
        self.assertEqual(state["status"], "in_progress")
        self.assertEqual(state["board"]["current_player"], 1)
        self.assertEqual(state["next_player_side"], "p1")
        self.assertGreater(len(state["legal_moves"]), 0)

        outsider_get = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_get.status_code, 403)

        outsider_state = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_state.status_code, 403)

    def test_submit_move_validates_turn_and_legality(self) -> None:
        p1 = self._register_and_login("match-p1b", "match-p1b@example.com")
        p2 = self._register_and_login("match-p2b", "match-p2b@example.com")
        outsider = self._register_and_login("match-out", "match-out@example.com")

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "player1_id": p1["user_id"],
                "player2_id": p2["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        spoof_resp = self.client.post(
            "/api/v1/matches",
            json={
                "player1_id": p2["user_id"],
                "player2_id": p1["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(spoof_resp.status_code, 403)

        first_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(first_move.status_code, 201)
        first = first_move.json()
        self.assertEqual(first["ply"], 0)
        self.assertEqual(first["player_side"], "p1")

        outsider_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 6, "c1": 0, "r2": 5, "c2": 1},
            },
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_move.status_code, 403)

        wrong_turn = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 0, "c1": 6, "r2": 1, "c2": 5},
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(wrong_turn.status_code, 403)

        illegal_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 6, "c1": 0, "r2": 6, "c2": 0},
            },
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(illegal_move.status_code, 400)

        illegal_pass = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={"pass_turn": True},
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(illegal_pass.status_code, 400)

    def test_admin_can_read_match_http_end_to_end(self) -> None:
        p1 = self._register_and_login("match-ap1", "match-ap1@example.com")
        p2 = self._register_and_login("match-ap2", "match-ap2@example.com")
        outsider = self._register_and_login("match-aout", "match-aout@example.com")
        admin = self._register_and_login("match-admin", "match-admin@example.com")

        self._promote_user_to_admin(admin["user_id"])

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "player1_id": p1["user_id"],
                "player2_id": p2["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        outsider_get = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_get.status_code, 403)

        outsider_state = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_state.status_code, 403)

        admin_get = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(admin_get.status_code, 200)
        self.assertEqual(admin_get.json()["id"], game_id)

        admin_state = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(admin_state.status_code, 200)
        self.assertEqual(admin_state.json()["game_id"], game_id)

    def _register_and_login(self, username: str, email: str) -> dict[str, str]:
        reg = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "supersecret123",
            },
        )
        self.assertEqual(reg.status_code, 201)
        user_id = reg.json()["id"]

        login = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": username, "password": "supersecret123"},
        )
        self.assertEqual(login.status_code, 200)
        return {
            "user_id": user_id,
            "access_token": login.json()["access_token"],
        }

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
