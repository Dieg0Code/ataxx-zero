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


class TestApiRankingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "ranking_integration.db"
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

    def test_season_active_and_rating_bootstrap(self) -> None:
        admin = self._register_and_login("rk-admin", "rk-admin@example.com")
        self._promote_user_to_admin(admin["user_id"])

        missing_active = self.client.get("/api/v1/ranking/seasons/active")
        self.assertEqual(missing_active.status_code, 404)

        season_resp = self.client.post(
            "/api/v1/ranking/seasons",
            json={"name": "Ranking S1", "is_active": True},
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(season_resp.status_code, 201)
        season = season_resp.json()
        season_id = season["id"]

        active_resp = self.client.get("/api/v1/ranking/seasons/active")
        self.assertEqual(active_resp.status_code, 200)
        self.assertEqual(active_resp.json()["id"], season_id)

        user = self._register_and_login("rank-user", "rank-user@example.com")
        user_id = user["user_id"]

        rating_resp = self.client.get(f"/api/v1/ranking/ratings/{user_id}/{season_id}")
        self.assertEqual(rating_resp.status_code, 200)
        rating = rating_resp.json()
        self.assertEqual(rating["user_id"], user_id)
        self.assertEqual(rating["season_id"], season_id)
        self.assertEqual(rating["rating"], 1200.0)
        self.assertEqual(rating["games_played"], 0)

        user2 = self._register_and_login("rank-user-2", "rank-user-2@example.com")
        user2_id = user2["user_id"]
        _ = self.client.get(f"/api/v1/ranking/ratings/{user2_id}/{season_id}")

        recompute_resp = self.client.post(
            f"/api/v1/ranking/leaderboard/{season_id}/recompute?limit=10",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(recompute_resp.status_code, 200)
        leaderboard = recompute_resp.json()
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard[0]["rank"], 1)
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
