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
from api.db.session import get_session

del _models


class TestApiAuthIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "auth_integration.db"
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

    def test_register_login_me_refresh_logout(self) -> None:
        register_resp = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "auth-user",
                "email": "auth-user@example.com",
                "password": "supersecret123",
            },
        )
        self.assertEqual(register_resp.status_code, 201)
        created = register_resp.json()
        self.assertEqual(created["username"], "auth-user")

        duplicate_resp = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "auth-user",
                "email": "auth-user-2@example.com",
                "password": "supersecret123",
            },
        )
        self.assertEqual(duplicate_resp.status_code, 409)

        bad_login = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "auth-user", "password": "wrong"},
        )
        self.assertEqual(bad_login.status_code, 401)

        login_resp = self.client.post(
            "/api/v1/auth/login",
            json={"username_or_email": "auth-user", "password": "supersecret123"},
        )
        self.assertEqual(login_resp.status_code, 200)
        tokens = login_resp.json()
        self.assertIn("access_token", tokens)
        self.assertIn("refresh_token", tokens)

        me_resp = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        self.assertEqual(me_resp.status_code, 200)
        me = me_resp.json()
        self.assertEqual(me["username"], "auth-user")

        refresh_resp = self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        self.assertEqual(refresh_resp.status_code, 200)
        refreshed = refresh_resp.json()
        self.assertNotEqual(refreshed["access_token"], tokens["access_token"])
        self.assertNotEqual(refreshed["refresh_token"], tokens["refresh_token"])

        old_refresh_again = self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        self.assertEqual(old_refresh_again.status_code, 401)

        logout_resp = self.client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refreshed["refresh_token"]},
        )
        self.assertEqual(logout_resp.status_code, 204)

        refresh_after_logout = self.client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refreshed["refresh_token"]},
        )
        self.assertEqual(refresh_after_logout.status_code, 401)


if __name__ == "__main__":
    unittest.main()
