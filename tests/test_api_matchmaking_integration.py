from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.enums import AgentType, BotKind
from api.db.models import BotProfile, Season, User
from api.db.session import get_session

del _models


class TestApiMatchmakingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "matchmaking_integration.db"
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

    def setUp(self) -> None:
        self._disable_all_bot_profiles()
        self._deactivate_all_seasons()
        self._create_active_season()

    def test_queue_join_matches_bot_when_only_bot_candidate_exists(self) -> None:
        human = self._register_and_login("mm-human", "mm-human@example.com")
        bot_user_id = self._create_bot_user_with_profile(
            username="mm-bot-hard",
            email="mm-bot-hard@example.com",
            heuristic_level="hard",
        )

        join_resp = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(join_resp.status_code, 200)
        join_payload = join_resp.json()
        self.assertEqual(join_payload["status"], "matched")
        self.assertEqual(join_payload["matched_with"], "bot")
        self.assertIsNotNone(join_payload["game_id"])

        game_id = join_payload["game_id"]
        game_resp = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 200)
        game_payload = game_resp.json()
        self.assertEqual(game_payload["queue_type"], "ranked")
        self.assertTrue(game_payload["rated"])
        self.assertEqual(game_payload["player2_id"], bot_user_id)
        self.assertEqual(game_payload["player2_agent"], "heuristic")

    def test_queue_matches_two_humans_when_available(self) -> None:
        p1 = self._register_and_login("mm-p1", "mm-p1@example.com")
        p2 = self._register_and_login("mm-p2", "mm-p2@example.com")

        join_1 = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(join_1.status_code, 200)
        self.assertEqual(join_1.json()["status"], "waiting")

        join_2 = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(join_2.status_code, 200)
        self.assertEqual(join_2.json()["status"], "matched")
        self.assertEqual(join_2.json()["matched_with"], "human")
        self.assertIsNotNone(join_2.json()["game_id"])

        status_1 = self.client.get(
            "/api/v1/matchmaking/queue/status",
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(status_1.status_code, 200)
        status_payload = status_1.json()
        self.assertEqual(status_payload["status"], "matched")
        self.assertEqual(status_payload["game_id"], join_2.json()["game_id"])

    def test_queue_uses_eligible_bot_pool_when_alternatives_exist(self) -> None:
        human = self._register_and_login("mm-bot-rotate", "mm-bot-rotate@example.com")
        bot_a_id = self._create_bot_user_with_profile(
            username="mm-bot-a",
            email="mm-bot-a@example.com",
            heuristic_level="easy",
        )
        bot_b_id = self._create_bot_user_with_profile(
            username="mm-bot-b",
            email="mm-bot-b@example.com",
            heuristic_level="hard",
        )

        first_join = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(first_join.status_code, 200)
        first_game_id = first_join.json()["game_id"]
        self.assertIsNotNone(first_game_id)

        first_game_resp = self.client.get(
            f"/api/v1/matches/{first_game_id}",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(first_game_resp.status_code, 200)
        first_bot_id = first_game_resp.json()["player2_id"]
        self.assertIsNotNone(first_bot_id)

        second_join = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(second_join.status_code, 200)
        second_game_id = second_join.json()["game_id"]
        self.assertIsNotNone(second_game_id)

        second_game_resp = self.client.get(
            f"/api/v1/matches/{second_game_id}",
            headers={"Authorization": f"Bearer {human['access_token']}"},
        )
        self.assertEqual(second_game_resp.status_code, 200)
        second_bot_id = second_game_resp.json()["player2_id"]
        self.assertIsNotNone(second_bot_id)
        self.assertIn(first_bot_id, {bot_a_id, bot_b_id})
        self.assertIn(second_bot_id, {bot_a_id, bot_b_id})

    def test_queue_ws_streams_waiting_status(self) -> None:
        user = self._register_and_login("mm-ws", "mm-ws@example.com")

        join_resp = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(join_resp.status_code, 200)
        self.assertEqual(join_resp.json()["status"], "waiting")

        with self.client.websocket_connect(
            f"/api/v1/matchmaking/queue/ws?token={user['access_token']}"
        ) as websocket:
            subscribed = websocket.receive_json()
            self.assertEqual(subscribed["type"], "queue.subscribed")

            status_event = websocket.receive_json()
            self.assertEqual(status_event["type"], "queue.status")
            self.assertEqual(status_event["payload"]["status"], "waiting")

    def test_queue_ws_rejects_missing_token(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as ctx, self.client.websocket_connect(
            "/api/v1/matchmaking/queue/ws"
        ) as websocket:
            websocket.receive_json()
        self.assertEqual(ctx.exception.code, 4401)

    def test_accept_and_reject_matched_game_flow(self) -> None:
        p1 = self._register_and_login("mm-accept-p1", "mm-accept-p1@example.com")
        p2 = self._register_and_login("mm-accept-p2", "mm-accept-p2@example.com")

        join_1 = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(join_1.status_code, 200)
        self.assertEqual(join_1.json()["status"], "waiting")

        join_2 = self.client.post(
            "/api/v1/matchmaking/queue/join",
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(join_2.status_code, 200)
        self.assertEqual(join_2.json()["status"], "matched")
        game_id = join_2.json()["game_id"]
        self.assertIsNotNone(game_id)

        accept_2 = self.client.post(
            "/api/v1/matchmaking/queue/accept",
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(accept_2.status_code, 200)
        self.assertEqual(accept_2.json()["decision"], "accepted")
        self.assertEqual(accept_2.json()["game_id"], game_id)

        reject_2 = self.client.post(
            "/api/v1/matchmaking/queue/reject",
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(reject_2.status_code, 200)
        self.assertEqual(reject_2.json()["decision"], "rejected")
        self.assertEqual(reject_2.json()["status"], "canceled")
        self.assertIsNone(reject_2.json()["game_id"])

        status_1 = self.client.get(
            "/api/v1/matchmaking/queue/status",
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(status_1.status_code, 200)
        self.assertIn(status_1.json()["status"], {"waiting", "matched"})

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
                    bot_kind=BotKind.HEURISTIC,
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

    def _create_active_season(self) -> None:
        async def _run() -> None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            async with self.sessionmaker() as session:
                season = Season(
                    name=f"Season MM {now.timestamp()}",
                    starts_at=now - timedelta(days=1),
                    ends_at=now + timedelta(days=30),
                    is_active=True,
                )
                session.add(season)
                await session.commit()

        asyncio.run(_run())

    def _deactivate_all_seasons(self) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                rows = await session.execute(select(Season))
                for season in rows.scalars().all():
                    season.is_active = False
                    session.add(season)
                await session.commit()

        asyncio.run(_run())

    def _disable_all_bot_profiles(self) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                rows = await session.execute(select(BotProfile))
                for profile in rows.scalars().all():
                    profile.enabled = False
                    session.add(profile)
                await session.commit()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
