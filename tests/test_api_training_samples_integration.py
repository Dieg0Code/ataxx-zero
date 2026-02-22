from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.session import get_session
from api.deps.inference import get_inference_service_dep
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.serialization import board_to_state
from inference.service import InferenceResult

del _models


class _StubInferenceService:
    def predict(self, board: AtaxxBoard, mode: str = "fast") -> InferenceResult:
        if board.is_game_over():
            return InferenceResult(
                move=None,
                action_idx=ACTION_SPACE.pass_index,
                value=1.0,
                mode="fast" if mode == "fast" else "strong",
            )
        move = (0, 0, 1, 1)
        return InferenceResult(
            move=move,
            action_idx=ACTION_SPACE.encode(move),
            value=0.2,
            mode="fast" if mode == "fast" else "strong",
        )


class TestApiTrainingSamplesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "training_samples_integration.db"
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
        app.dependency_overrides[get_inference_service_dep] = _StubInferenceService
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

        async def _dispose() -> None:
            await cls.engine.dispose()

        asyncio.run(_dispose())
        cls.tmpdir.cleanup()

    def test_create_and_fetch_training_sample(self) -> None:
        user = self._register_and_login("ts-u1", "ts-u1@example.com")
        game_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 201)
        game_id = game_resp.json()["id"]

        create_resp = self.client.post(
            "/api/v1/training/samples",
            json={
                "game_id": game_id,
                "ply": 0,
                "player_side": "p1",
                "observation": {"grid": [[0] * 7 for _ in range(7)], "current_player": 1},
                "policy_target": {"10": 1.0},
                "value_target": 1.0,
                "sample_weight": 1.0,
                "split": "train",
                "source": "self_play",
            },
        )
        self.assertEqual(create_resp.status_code, 201)
        sample = create_resp.json()
        sample_id = sample["id"]
        self.assertEqual(sample["game_id"], game_id)
        self.assertEqual(sample["policy_target"]["10"], 1.0)

        get_resp = self.client.get(f"/api/v1/training/samples/{sample_id}")
        self.assertEqual(get_resp.status_code, 200)
        fetched = get_resp.json()
        self.assertEqual(fetched["id"], sample_id)
        self.assertEqual(fetched["split"], "train")

    def test_ingest_finished_game_moves(self) -> None:
        user = self._register_and_login("ts-u2", "ts-u2@example.com")
        game_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 201)
        game_id = game_resp.json()["id"]

        terminal_board = AtaxxBoard()
        terminal_board.grid[:, :] = 1
        terminal_board.current_player = 1

        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(terminal_board), "mode": "fast"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)

        ingest_resp = self.client.post(
            f"/api/v1/training/samples/ingest-game/{game_id}?split=train&source=self_play&overwrite=true"
        )
        self.assertEqual(ingest_resp.status_code, 200)
        payload = ingest_resp.json()
        self.assertEqual(payload["game_id"], game_id)
        self.assertEqual(payload["created_count"], 1)

        list_resp = self.client.get(f"/api/v1/training/samples?game_id={game_id}&split=train")
        self.assertEqual(list_resp.status_code, 200)
        page = list_resp.json()
        samples = page["items"]
        self.assertEqual(page["total"], 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["value_target"], 1.0)
        self.assertEqual(samples[0]["player_side"], "p1")

        stats_resp = self.client.get(f"/api/v1/training/samples/stats?game_id={game_id}")
        self.assertEqual(stats_resp.status_code, 200)
        stats = stats_resp.json()
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["by_split"]["train"], 1)
        self.assertEqual(stats["by_source"]["self_play"], 1)

        export_resp = self.client.get(
            f"/api/v1/training/samples/export.ndjson?game_id={game_id}&split=train&limit=10"
        )
        self.assertEqual(export_resp.status_code, 200)
        self.assertEqual(export_resp.headers["content-type"].split(";")[0], "text/plain")
        lines = [line for line in export_resp.text.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn('"game_id"', lines[0])
        self.assertIn('"policy_target"', lines[0])

        npz_resp = self.client.get(
            f"/api/v1/training/samples/export.npz?game_id={game_id}&split=train&limit=10"
        )
        self.assertEqual(npz_resp.status_code, 200)
        self.assertEqual(
            npz_resp.headers["content-type"].split(";")[0],
            "application/octet-stream",
        )
        npz_file = np.load(io.BytesIO(npz_resp.content))
        self.assertIn("sample_ids", npz_file.files)
        self.assertIn("value_targets", npz_file.files)
        self.assertEqual(len(npz_file["sample_ids"]), 1)
        self.assertEqual(float(npz_file["value_targets"][0]), 1.0)

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


if __name__ == "__main__":
    unittest.main()
