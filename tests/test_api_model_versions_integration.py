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


class TestApiModelVersionsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "model_versions_integration.db"
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

    def test_create_list_get_model_versions(self) -> None:
        create_resp = self.client.post(
            "/api/v1/model-versions",
            json={
                "name": "ataxx-v1",
                "hf_repo_id": "dieg0code/ataxx-zero",
                "checkpoint_uri": "hf://dieg0code/ataxx-zero/model_iter_006.pt",
                "onnx_uri": "hf://dieg0code/ataxx-zero/ataxx_model.onnx",
                "notes": "baseline model",
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.json()
        version_id = created["id"]
        self.assertEqual(created["name"], "ataxx-v1")
        self.assertFalse(created["is_active"])

        get_resp = self.client.get(f"/api/v1/model-versions/{version_id}")
        self.assertEqual(get_resp.status_code, 200)
        fetched = get_resp.json()
        self.assertEqual(fetched["id"], version_id)
        self.assertEqual(fetched["name"], "ataxx-v1")

        list_resp = self.client.get("/api/v1/model-versions?limit=10")
        self.assertEqual(list_resp.status_code, 200)
        page = list_resp.json()
        rows = page["items"]
        self.assertGreaterEqual(len(rows), 1)
        ids = {row["id"] for row in rows}
        self.assertIn(version_id, ids)

    def test_unique_name_and_single_active_rule(self) -> None:
        create_v2 = self.client.post(
            "/api/v1/model-versions",
            json={"name": "ataxx-v2", "is_active": True},
        )
        self.assertEqual(create_v2.status_code, 201)
        v2 = create_v2.json()
        self.assertTrue(v2["is_active"])

        dup_resp = self.client.post(
            "/api/v1/model-versions",
            json={"name": "ataxx-v2"},
        )
        self.assertEqual(dup_resp.status_code, 409)

        create_v3 = self.client.post(
            "/api/v1/model-versions",
            json={"name": "ataxx-v3", "is_active": True},
        )
        self.assertEqual(create_v3.status_code, 201)
        v3 = create_v3.json()
        self.assertTrue(v3["is_active"])

        active_resp = self.client.get("/api/v1/model-versions/active")
        self.assertEqual(active_resp.status_code, 200)
        active = active_resp.json()
        self.assertEqual(active["id"], v3["id"])

        activate_v2 = self.client.post(f"/api/v1/model-versions/{v2['id']}/activate")
        self.assertEqual(activate_v2.status_code, 200)
        activated = activate_v2.json()
        self.assertEqual(activated["id"], v2["id"])
        self.assertTrue(activated["is_active"])

        active_after = self.client.get("/api/v1/model-versions/active")
        self.assertEqual(active_after.status_code, 200)
        self.assertEqual(active_after.json()["id"], v2["id"])

        v3_after = self.client.get(f"/api/v1/model-versions/{v3['id']}")
        self.assertEqual(v3_after.status_code, 200)
        self.assertFalse(v3_after.json()["is_active"])


if __name__ == "__main__":
    unittest.main()
