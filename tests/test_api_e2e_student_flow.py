from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.enums import AgentType
from api.db.models import BotProfile, User
from api.db.session import get_session

del _models


class TestApiE2EStudentFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "e2e_student_flow.db"
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

    def test_student_flow_register_play_replay_delete(self) -> None:
        player = self._register_and_login("e2e-student", "e2e-student@example.com")
        bot_user_id = self._create_bot_user_with_profile(
            username="e2e-bot-normal",
            email="e2e-bot-normal@example.com",
            heuristic_level="normal",
        )

        create_match = self.client.post(
            "/api/v1/matches",
            json={
                "queue_type": "vs_ai",
                "rated": False,
                "player2_id": bot_user_id,
                "player1_agent": "human",
                "player2_agent": "human",
            },
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(create_match.status_code, 201)
        game_id = create_match.json()["id"]

        state_before = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(state_before.status_code, 200)
        legal_moves = state_before.json()["legal_moves"]
        self.assertGreater(len(legal_moves), 0)
        first_move = legal_moves[0]

        move_resp = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={"pass_turn": False, "move": first_move},
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)
        self.assertEqual(move_resp.json()["player_side"], "p1")

        bot_resp = self.client.post(
            f"/api/v1/matches/{game_id}/advance-bot",
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(bot_resp.status_code, 200)
        self.assertTrue(bot_resp.json()["applied"])

        replay_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}/replay",
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay_payload = replay_resp.json()
        self.assertEqual(replay_payload["game"]["id"], game_id)
        self.assertGreaterEqual(len(replay_payload["moves"]), 2)

        delete_resp = self.client.delete(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(delete_resp.status_code, 204)

        get_deleted = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {player['access_token']}"},
        )
        self.assertEqual(get_deleted.status_code, 404)

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

    def _create_bot_user_with_profile(
        self,
        *,
        username: str,
        email: str,
        heuristic_level: str,
    ) -> str:
        async def _run() -> str:
            async with self.sessionmaker() as session:
                user = User(
                    username=username,
                    email=email,
                    password_hash=None,
                    is_active=True,
                    is_admin=False,
                    is_bot=True,
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
                profile = BotProfile(
                    user_id=user.id,
                    agent_type=AgentType.HEURISTIC,
                    heuristic_level=heuristic_level,
                    model_mode=None,
                    enabled=True,
                )
                session.add(profile)
                await session.commit()
                return str(user.id)

        return asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
