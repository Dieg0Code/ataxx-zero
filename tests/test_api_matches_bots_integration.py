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


class TestApiMatchesBotsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "matches_bots_integration.db"
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

    def test_create_match_vs_bot_derives_agent_and_advances_bot_turn(self) -> None:
        human = self._register_and_login("botmatch-human", "botmatch-human@example.com")
        bot_user_id = self._create_bot_user_with_profile(
            username="ub_ai_normal",
            email="ub_ai_normal@example.com",
            agent_type=AgentType.HEURISTIC,
            heuristic_level="normal",
            model_mode=None,
        )

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "queue_type": "vs_ai",
                "rated": False,
                "player2_id": bot_user_id,
                "player1_agent": "human",
                "player2_agent": "human",
            },
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.json()
        self.assertEqual(created["player2_id"], bot_user_id)
        self.assertEqual(created["player2_agent"], "heuristic")

        game_id = created["id"]
        first_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
            },
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(first_move.status_code, 201)
        self.assertEqual(first_move.json()["player_side"], "p1")

        advance_resp = self.client.post(
            f"/api/v1/matches/{game_id}/advance-bot",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(advance_resp.status_code, 200)
        payload = advance_resp.json()
        self.assertTrue(payload["applied"])
        self.assertEqual(payload["game_id"], game_id)
        self.assertIsNotNone(payload["move"])
        self.assertEqual(payload["move"]["player_side"], "p2")
        self.assertEqual(payload["move"]["mode"], "heuristic_normal")

    def test_create_ranked_match_vs_bot_is_allowed(self) -> None:
        human = self._register_and_login("botrank-human", "botrank-human@example.com")
        bot_user_id = self._create_bot_user_with_profile(
            username="ub_ai_ranked",
            email="ub_ai_ranked@example.com",
            agent_type=AgentType.HEURISTIC,
            heuristic_level="hard",
            model_mode=None,
        )
        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "queue_type": "ranked",
                "rated": True,
                "player2_id": bot_user_id,
                "player1_agent": "human",
                "player2_agent": "human",
            },
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.json()
        self.assertEqual(created["player2_agent"], "heuristic")
        self.assertTrue(created["rated"])

    def test_identity_bots_lists_only_enabled_profiles(self) -> None:
        user = self._register_and_login("botlist-user", "botlist-user@example.com")
        self._create_bot_user_with_profile(
            username="ub_ai_easy",
            email="ub_ai_easy@example.com",
            agent_type=AgentType.HEURISTIC,
            heuristic_level="easy",
            model_mode=None,
            enabled=True,
        )
        self._create_bot_user_with_profile(
            username="ub_ai_disabled",
            email="ub_ai_disabled@example.com",
            agent_type=AgentType.HEURISTIC,
            heuristic_level="hard",
            model_mode=None,
            enabled=False,
        )

        resp = self.client.get(
            "/api/v1/identity/bots?limit=20&offset=0",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(resp.status_code, 200)
        page = resp.json()
        usernames = {item["username"] for item in page["items"]}
        self.assertIn("ub_ai_easy", usernames)
        self.assertNotIn("ub_ai_disabled", usernames)

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
        agent_type: AgentType,
        heuristic_level: str | None,
        model_mode: str | None,
        enabled: bool = True,
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
                    agent_type=agent_type,
                    heuristic_level=heuristic_level,
                    model_mode=model_mode,
                    enabled=enabled,
                )
                session.add(profile)
                await session.commit()
                return str(user.id)

        return asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
