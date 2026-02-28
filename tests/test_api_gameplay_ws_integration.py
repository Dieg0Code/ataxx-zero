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
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.session import get_session
from game.board import AtaxxBoard
from game.serialization import board_to_state

del _models


class TestApiGameplayWsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "gameplay_ws_integration.db"
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

    def test_gameplay_ws_streams_move_events(self) -> None:
        user = self._register_and_login("ws-user", "ws-user@example.com")

        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        with self.client.websocket_connect(
            f"/api/v1/gameplay/games/{game_id}/ws?token={user['access_token']}"
        ) as websocket:
            subscribed = websocket.receive_json()
            self.assertEqual(subscribed["type"], "game.subscribed")
            self.assertEqual(subscribed["game_id"], game_id)

            board = AtaxxBoard()
            move_resp = self.client.post(
                f"/api/v1/gameplay/games/{game_id}/move/manual",
                json={
                    "board": board_to_state(board),
                    "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
                    "mode": "manual",
                },
                headers={"Authorization": f"Bearer {user['access_token']}"},
            )
            self.assertEqual(move_resp.status_code, 201)

            event = websocket.receive_json()
            self.assertEqual(event["type"], "game.move.applied")
            self.assertEqual(event["game_id"], game_id)
            self.assertEqual(event["move"]["ply"], 0)
            self.assertEqual(event["move"]["r1"], 0)
            self.assertEqual(event["move"]["r2"], 1)
            self.assertIn(event["game"]["status"], {"pending", "in_progress"})

    def test_gameplay_ws_rejects_missing_token(self) -> None:
        user = self._register_and_login("ws-user-2", "ws-user-2@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        with self.assertRaises(WebSocketDisconnect) as ctx, self.client.websocket_connect(
            f"/api/v1/gameplay/games/{game_id}/ws"
        ) as websocket:
            websocket.receive_json()
        self.assertEqual(ctx.exception.code, 4401)

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
        return {
            "user_id": user_id,
            "access_token": login.json()["access_token"],
        }


if __name__ == "__main__":
    unittest.main()
