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


class TestApiIdentityIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "identity_integration.db"
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

    def test_identity_permissions_and_conflicts(self) -> None:
        admin = self._register_and_login("id-admin", "id-admin@example.com")
        self._promote_user_to_admin(admin["user_id"])

        user_a = self._register_and_login("id-user-a", "id-user-a@example.com")
        user_b = self._register_and_login("id-user-b", "id-user-b@example.com")

        create_by_admin = self.client.post(
            "/api/v1/identity/users",
            json={
                "username": "seeded-user",
                "email": "seeded@example.com",
                "is_bot": False,
            },
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(create_by_admin.status_code, 201)
        seeded = create_by_admin.json()
        seeded_id = seeded["id"]

        create_by_normal = self.client.post(
            "/api/v1/identity/users",
            json={"username": "nope-user", "email": "nope@example.com"},
            headers={"Authorization": f"Bearer {user_a['access_token']}"},
        )
        self.assertEqual(create_by_normal.status_code, 403)

        get_self = self.client.get(
            f"/api/v1/identity/users/{user_a['user_id']}",
            headers={"Authorization": f"Bearer {user_a['access_token']}"},
        )
        self.assertEqual(get_self.status_code, 200)
        self.assertEqual(get_self.json()["id"], user_a["user_id"])

        get_other_forbidden = self.client.get(
            f"/api/v1/identity/users/{user_b['user_id']}",
            headers={"Authorization": f"Bearer {user_a['access_token']}"},
        )
        self.assertEqual(get_other_forbidden.status_code, 403)

        get_other_admin = self.client.get(
            f"/api/v1/identity/users/{user_b['user_id']}",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(get_other_admin.status_code, 200)
        self.assertEqual(get_other_admin.json()["id"], user_b["user_id"])

        list_forbidden = self.client.get(
            "/api/v1/identity/users?limit=10",
            headers={"Authorization": f"Bearer {user_a['access_token']}"},
        )
        self.assertEqual(list_forbidden.status_code, 403)

        list_admin = self.client.get(
            "/api/v1/identity/users?limit=10",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(list_admin.status_code, 200)
        page = list_admin.json()
        self.assertGreaterEqual(page["total"], 3)
        ids = {row["id"] for row in page["items"]}
        self.assertIn(seeded_id, ids)
        self.assertIn(user_a["user_id"], ids)
        self.assertIn(user_b["user_id"], ids)

        conflict_resp = self.client.post(
            "/api/v1/identity/users",
            json={"username": "seeded-user", "email": "another@example.com"},
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(conflict_resp.status_code, 409)

    def test_get_user_not_found_for_admin(self) -> None:
        admin = self._register_and_login("id-admin2", "id-admin2@example.com")
        self._promote_user_to_admin(admin["user_id"])
        missing = "00000000-0000-0000-0000-000000000001"
        resp = self.client.get(
            f"/api/v1/identity/users/{missing}",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(resp.status_code, 404)
        self.assertIn("User not found", resp.json()["detail"])

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
