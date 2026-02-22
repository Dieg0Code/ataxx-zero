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
from api.config import Settings
from api.db import models as _models
from api.db.session import get_session

del _models


class TestApiAuthRateLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "auth_rate_limit.db"
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
        settings = Settings(
            app_env="test",
            app_docs_enabled=False,
            database_url=cls.database_url,
            auth_rate_limit_enabled=True,
            auth_login_rate_limit_requests=2,
            auth_login_rate_limit_window_s=60,
            auth_refresh_rate_limit_requests=1,
            auth_refresh_rate_limit_window_s=60,
        )
        app = create_app(settings=settings)

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

    def test_login_rate_limit_returns_429(self) -> None:
        register_resp = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "rl-user",
                "email": "rl-user@example.com",
                "password": "supersecret123",
            },
        )
        self.assertEqual(register_resp.status_code, 201)

        bad1 = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "rl-user", "password": "wrong"},
        )
        bad2 = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "rl-user", "password": "wrong"},
        )
        blocked = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "rl-user", "password": "wrong"},
        )

        self.assertEqual(bad1.status_code, 401)
        self.assertEqual(bad2.status_code, 401)
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Retry-After", blocked.headers)
        payload = blocked.json()
        self.assertEqual(payload["error_code"], "too_many_requests")

    def test_refresh_rate_limit_returns_429(self) -> None:
        register_resp = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "rl-user-refresh",
                "email": "rl-user-refresh@example.com",
                "password": "supersecret123",
            },
        )
        self.assertEqual(register_resp.status_code, 201)

        login_resp = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "rl-user-refresh", "password": "supersecret123"},
        )
        self.assertEqual(login_resp.status_code, 200)
        refresh_token = login_resp.json()["refresh_token"]

        first = self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        second = self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)
        payload = second.json()
        self.assertEqual(payload["error_code"], "too_many_requests")


if __name__ == "__main__":
    unittest.main()
