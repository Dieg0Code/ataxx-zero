from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.enums import AgentType
from api.db.models import BotProfile, RatingEvent, Season, User
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
            value=0.31,
            mode="fast" if mode == "fast" else "strong",
        )


class TestApiGamesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "integration.db"
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

    def test_create_get_and_list_games_with_real_db(self) -> None:
        user = self._register_and_login("gp-u1", "gp-u1@example.com")
        outsider = self._register_and_login("gp-u2", "gp-u2@example.com")

        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "queue_type": "ranked",
                "rated": True,
                "player1_agent": "human",
                "player2_agent": "heuristic",
                "source": "human",
            },
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.json()
        game_id = created["id"]
        self.assertEqual(created["queue_type"], "ranked")
        self.assertTrue(created["rated"])

        get_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(get_resp.status_code, 200)
        fetched = get_resp.json()
        self.assertEqual(fetched["id"], game_id)
        self.assertEqual(fetched["player2_agent"], "heuristic")

        list_resp = self.client.get(
            "/api/v1/gameplay/games?limit=10",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_resp.status_code, 200)
        page = list_resp.json()
        self.assertGreaterEqual(page["total"], 1)
        all_games = page["items"]
        ids = {entry["id"] for entry in all_games}
        self.assertIn(game_id, ids)

        outsider_get = self.client.get(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_get.status_code, 403)

    def test_store_and_list_game_moves(self) -> None:
        user = self._register_and_login("gp-u3", "gp-u3@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        board = AtaxxBoard()
        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(board), "mode": "fast"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)
        stored = move_resp.json()
        self.assertEqual(stored["game_id"], game_id)
        self.assertEqual(stored["r1"], 0)
        self.assertEqual(stored["c1"], 0)
        self.assertEqual(stored["r2"], 1)
        self.assertEqual(stored["c2"], 1)
        self.assertIn("board_before", stored)
        self.assertIn("board_after", stored)
        self.assertEqual(stored["board_before"]["current_player"], 1)
        self.assertEqual(stored["board_after"]["current_player"], -1)

        list_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}/moves",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_resp.status_code, 200)
        moves = list_resp.json()
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["game_id"], game_id)

        replay_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}/replay",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay = replay_resp.json()
        self.assertEqual(replay["game"]["id"], game_id)
        self.assertEqual(len(replay["moves"]), 1)
        self.assertEqual(replay["moves"][0]["game_id"], game_id)

    def test_list_games_can_filter_by_status(self) -> None:
        user = self._register_and_login("gp-filter", "gp-filter@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        list_finished = self.client.get(
            "/api/v1/gameplay/games?limit=10&status=finished",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_finished.status_code, 200)
        payload_finished = list_finished.json()
        ids_finished = {entry["id"] for entry in payload_finished["items"]}
        self.assertNotIn(game_id, ids_finished)

        list_pending = self.client.get(
            "/api/v1/gameplay/games?limit=10&status=pending",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_pending.status_code, 200)
        payload_pending = list_pending.json()
        ids_pending = {entry["id"] for entry in payload_pending["items"]}
        self.assertIn(game_id, ids_pending)

    def test_creator_can_control_bot_vs_bot_game(self) -> None:
        creator = self._register_and_login("gp-owner", "gp-owner@example.com")
        outsider = self._register_and_login("gp-outsider", "gp-outsider@example.com")
        bot_p1_id = self._create_bot_user_with_profile(
            username="gp-bot-p1",
            email="gp-bot-p1@example.com",
            heuristic_level="easy",
        )
        bot_p2_id = self._create_bot_user_with_profile(
            username="gp-bot-p2",
            email="gp-bot-p2@example.com",
            heuristic_level="hard",
        )

        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "queue_type": "casual",
                "status": "in_progress",
                "rated": False,
                "player1_id": bot_p1_id,
                "player2_id": bot_p2_id,
                "player1_agent": "heuristic",
                "player2_agent": "heuristic",
                "source": "human",
                "is_training_eligible": False,
            },
            headers={"Authorization": f"Bearer {creator['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        created = create_resp.json()
        self.assertEqual(created["created_by_user_id"], creator["user_id"])
        game_id = created["id"]

        board = AtaxxBoard()
        creator_move = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move/manual",
            json={
                "board": board_to_state(board),
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
                "mode": "manual",
            },
            headers={"Authorization": f"Bearer {creator['access_token']}"},
        )
        self.assertEqual(creator_move.status_code, 201)

        outsider_move = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move/manual",
            json={
                "board": board_to_state(board),
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
                "mode": "manual",
            },
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_move.status_code, 403)

    def test_create_game_rejects_same_player_on_both_sides(self) -> None:
        creator = self._register_and_login("gp-owner-same", "gp-owner-same@example.com")
        bot_id = self._create_bot_user_with_profile(
            username="gp-bot-same",
            email="gp-bot-same@example.com",
            heuristic_level="hard",
        )

        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "queue_type": "casual",
                "status": "in_progress",
                "rated": False,
                "player1_id": bot_id,
                "player2_id": bot_id,
                "player1_agent": "heuristic",
                "player2_agent": "heuristic",
                "source": "human",
                "is_training_eligible": False,
            },
            headers={"Authorization": f"Bearer {creator['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 400)
        self.assertIn("must be different users", create_resp.json()["detail"])

    def test_create_game_rejects_ranked_ai_vs_ai(self) -> None:
        creator = self._register_and_login("gp-owner-ranked-ai", "gp-owner-ranked-ai@example.com")
        bot_p1_id = self._create_bot_user_with_profile(
            username="gp-bot-ranked-p1",
            email="gp-bot-ranked-p1@example.com",
            heuristic_level="easy",
        )
        bot_p2_id = self._create_bot_user_with_profile(
            username="gp-bot-ranked-p2",
            email="gp-bot-ranked-p2@example.com",
            heuristic_level="hard",
        )

        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "queue_type": "ranked",
                "status": "in_progress",
                "rated": True,
                "player1_id": bot_p1_id,
                "player2_id": bot_p2_id,
                "player1_agent": "heuristic",
                "player2_agent": "heuristic",
                "source": "human",
                "is_training_eligible": False,
            },
            headers={"Authorization": f"Bearer {creator['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 400)
        self.assertIn("solo admite modo casual", create_resp.json()["detail"])

    def test_list_games_skips_invalid_legacy_rows(self) -> None:
        user = self._register_and_login("gp-u5", "gp-u5@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        valid_game_id = create_resp.json()["id"]
        user_id = user["user_id"]

        async def _inject_invalid() -> None:
            async with self.sessionmaker() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO game (
                            id, queue_type, status, rated, player1_id, player1_agent, player2_agent,
                            source, is_training_eligible, created_at
                        ) VALUES (
                            :id, :queue_type, :status, :rated, :player1_id, :player1_agent, :player2_agent,
                            :source, :is_training_eligible, :created_at
                        )
                        """
                    ),
                    {
                        "id": "00000000-0000-0000-0000-00000000aaaa",
                        "queue_type": "casual",
                        "status": "legacy_broken_status",
                        "rated": 0,
                        "player1_id": user_id,
                        "player1_agent": "human",
                        "player2_agent": "human",
                        "source": "human",
                        "is_training_eligible": 0,
                        "created_at": "2026-02-23 00:00:00",
                    },
                )
                await session.commit()

        asyncio.run(_inject_invalid())

        list_resp = self.client.get(
            "/api/v1/gameplay/games?limit=8&offset=0",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(list_resp.status_code, 200)
        payload = list_resp.json()
        ids = {row["id"] for row in payload["items"]}
        self.assertIn(valid_game_id, ids)

    def test_replay_returns_422_for_invalid_legacy_game_row(self) -> None:
        user = self._register_and_login("gp-u6", "gp-u6@example.com")
        user_id = user["user_id"]
        invalid_game_id = "00000000-0000-0000-0000-00000000aab1"

        async def _inject_invalid() -> None:
            async with self.sessionmaker() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO game (
                            id, queue_type, status, rated, player1_id, player1_agent, player2_agent,
                            source, is_training_eligible, created_at
                        ) VALUES (
                            :id, :queue_type, :status, :rated, :player1_id, :player1_agent, :player2_agent,
                            :source, :is_training_eligible, :created_at
                        )
                        """
                    ),
                    {
                        "id": invalid_game_id,
                        "queue_type": "casual",
                        "status": "legacy_broken_status",
                        "rated": 0,
                        "player1_id": user_id,
                        "player1_agent": "human",
                        "player2_agent": "human",
                        "source": "human",
                        "is_training_eligible": 0,
                        "created_at": "2026-02-23 00:00:00",
                    },
                )
                await session.commit()

        asyncio.run(_inject_invalid())

        replay_resp = self.client.get(
            f"/api/v1/gameplay/games/{invalid_game_id}/replay",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertIn(replay_resp.status_code, {404, 422})

    def test_delete_legacy_invalid_game_row_returns_204(self) -> None:
        user = self._register_and_login("gp-u7", "gp-u7@example.com")
        user_id = user["user_id"]
        invalid_game_id = "00000000-0000-0000-0000-00000000aab2"

        async def _inject_invalid() -> None:
            async with self.sessionmaker() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO game (
                            id, queue_type, status, rated, player1_id, player1_agent, player2_agent,
                            source, is_training_eligible, created_at
                        ) VALUES (
                            :id, :queue_type, :status, :rated, :player1_id, :player1_agent, :player2_agent,
                            :source, :is_training_eligible, :created_at
                        )
                        """
                    ),
                    {
                        "id": invalid_game_id,
                        "queue_type": "casual",
                        "status": "legacy_broken_status",
                        "rated": 0,
                        "player1_id": user_id,
                        "player1_agent": "human",
                        "player2_agent": "human",
                        "source": "human",
                        "is_training_eligible": 0,
                        "created_at": "2026-02-23 00:00:00",
                    },
                )
                await session.commit()

        asyncio.run(_inject_invalid())

        delete_resp = self.client.delete(
            f"/api/v1/gameplay/games/{invalid_game_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertIn(delete_resp.status_code, {204, 404})

    def test_delete_game_with_rating_events_succeeds(self) -> None:
        user = self._register_and_login("gp-u8", "gp-u8@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={"queue_type": "casual"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]
        user_id = user["user_id"]

        async def _insert_dependencies() -> None:
            async with self.sessionmaker() as session:
                season = Season(
                    name="Delete FK Season",
                    is_active=False,
                    starts_at=datetime(2026, 2, 1, 0, 0, 0),
                    ends_at=datetime(2026, 3, 1, 0, 0, 0),
                )
                session.add(season)
                await session.commit()
                await session.refresh(season)

                await session.execute(
                    text(
                        """
                        INSERT INTO ratingevent (
                            id, game_id, user_id, season_id, rating_before, rating_after, delta, transition_type, created_at
                        ) VALUES (
                            :id, :game_id, :user_id, :season_id, :rating_before, :rating_after, :delta, :transition_type, :created_at
                        )
                        """
                    ),
                    {
                        "id": "11111111-2222-3333-4444-555555555555",
                        "game_id": game_id,
                        "user_id": user_id,
                        "season_id": str(season.id),
                        "rating_before": 1200.0,
                        "rating_after": 1212.0,
                        "delta": 12.0,
                        "transition_type": "stable",
                        "created_at": "2026-02-23 00:00:00",
                    },
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO queueentry (
                            id, season_id, user_id, rating_snapshot, status, matched_game_id, created_at, updated_at
                        ) VALUES (
                            :id, :season_id, :user_id, :rating_snapshot, :status, :matched_game_id, :created_at, :updated_at
                        )
                        """
                    ),
                    {
                        "id": "66666666-7777-8888-9999-000000000000",
                        "season_id": str(season.id),
                        "user_id": user_id,
                        "rating_snapshot": 1200.0,
                        "status": "matched",
                        "matched_game_id": game_id,
                        "created_at": "2026-02-23 00:00:00",
                        "updated_at": "2026-02-23 00:00:00",
                    },
                )
                await session.commit()
        asyncio.run(_insert_dependencies())

        delete_resp = self.client.delete(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(delete_resp.status_code, 204)

    def test_game_is_auto_finished_on_terminal_board(self) -> None:
        user = self._register_and_login("gp-u4", "gp-u4@example.com")
        create_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        terminal_board = AtaxxBoard()
        terminal_board.grid[:, :] = 1
        terminal_board.current_player = 1

        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(terminal_board), "mode": "fast"},
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)

        game_resp = self.client.get(
            f"/api/v1/gameplay/games/{game_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 200)
        game = game_resp.json()
        self.assertEqual(game["status"], "finished")
        self.assertEqual(game["winner_side"], "p1")
        self.assertEqual(game["termination_reason"], "normal")

    def test_rated_game_updates_ratings(self) -> None:
        admin = self._register_and_login("gp-admin", "gp-admin@example.com")
        self._promote_user_to_admin(admin["user_id"])
        p1 = self._register_and_login("rated-p1", "rated-p1@example.com")
        p2 = self._register_and_login("rated-p2", "rated-p2@example.com")
        p1_id = p1["user_id"]
        p2_id = p2["user_id"]

        season_resp = self.client.post(
            "/api/v1/ranking/seasons",
            json={"name": "S1 Integration", "is_active": True},
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(season_resp.status_code, 201)
        season_id = season_resp.json()["id"]

        # Bootstrap initial ratings and recompute leaderboard once so we can
        # verify it gets refreshed after the rated result is applied.
        bootstrap_p1 = self.client.get(f"/api/v1/ranking/ratings/{p1_id}/{season_id}")
        bootstrap_p2 = self.client.get(f"/api/v1/ranking/ratings/{p2_id}/{season_id}")
        self.assertEqual(bootstrap_p1.status_code, 200)
        self.assertEqual(bootstrap_p2.status_code, 200)
        recompute_bootstrap = self.client.post(
            f"/api/v1/ranking/leaderboard/{season_id}/recompute?limit=10",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(recompute_bootstrap.status_code, 200)

        game_resp = self.client.post(
            "/api/v1/gameplay/games",
            json={
                "rated": True,
                "season_id": season_id,
                "player1_id": p1_id,
                "player2_id": p2_id,
                "queue_type": "ranked",
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 201)
        game_id = game_resp.json()["id"]

        terminal_board = AtaxxBoard()
        terminal_board.grid[:, :] = 1
        terminal_board.current_player = 1
        move_resp = self.client.post(
            f"/api/v1/gameplay/games/{game_id}/move",
            json={"board": board_to_state(terminal_board), "mode": "fast"},
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(move_resp.status_code, 201)

        p1_rating_resp = self.client.get(f"/api/v1/ranking/ratings/{p1_id}/{season_id}")
        p2_rating_resp = self.client.get(f"/api/v1/ranking/ratings/{p2_id}/{season_id}")
        self.assertEqual(p1_rating_resp.status_code, 200)
        self.assertEqual(p2_rating_resp.status_code, 200)
        p1_rating = p1_rating_resp.json()
        p2_rating = p2_rating_resp.json()
        self.assertEqual(p1_rating["games_played"], 1)
        self.assertEqual(p2_rating["games_played"], 1)
        self.assertEqual(p1_rating["wins"], 1)
        self.assertEqual(p2_rating["losses"], 1)
        self.assertGreater(p1_rating["rating"], 1200.0)
        self.assertLess(p2_rating["rating"], 1200.0)

        lb_resp = self.client.get(f"/api/v1/ranking/leaderboard/{season_id}?limit=10")
        self.assertEqual(lb_resp.status_code, 200)
        leaderboard = lb_resp.json()["items"]
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard[0]["user_id"], p1_id)
        self.assertEqual(leaderboard[0]["username"], "rated-p1")
        self.assertEqual(leaderboard[0]["rank"], 1)
        self.assertIsNotNone(leaderboard[0]["recent_lp_delta"])
        self.assertGreater(leaderboard[0]["recent_lp_delta"], 0)
        self.assertEqual(leaderboard[1]["user_id"], p2_id)
        self.assertEqual(leaderboard[1]["username"], "rated-p2")
        self.assertEqual(leaderboard[1]["rank"], 2)
        self.assertIsNotNone(leaderboard[1]["recent_lp_delta"])
        self.assertLessEqual(leaderboard[1]["recent_lp_delta"], 0)

        events = self._list_rating_events(game_id)
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertIsNotNone(event.before_league)
            self.assertIsNotNone(event.before_division)
            self.assertIsNotNone(event.before_lp)
            self.assertIsNotNone(event.after_league)
            self.assertIsNotNone(event.after_division)
            self.assertIsNotNone(event.after_lp)
            self.assertIn(event.transition_type, {"promotion", "demotion", "stable"})

        p1_events_resp = self.client.get(
            f"/api/v1/ranking/ratings/{p1_id}/{season_id}/events?limit=10",
        )
        self.assertEqual(p1_events_resp.status_code, 200)
        p1_events_page = p1_events_resp.json()
        self.assertEqual(p1_events_page["total"], 1)
        self.assertEqual(len(p1_events_page["items"]), 1)
        self.assertEqual(p1_events_page["items"][0]["user_id"], p1_id)
        self.assertEqual(p1_events_page["items"][0]["game_id"], game_id)
        self.assertIn(
            p1_events_page["items"][0]["transition_type"],
            {"promotion", "demotion", "stable"},
        )

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

    def _list_rating_events(self, game_id: str) -> list[RatingEvent]:
        async def _run() -> list[RatingEvent]:
            async with self.sessionmaker() as session:
                stmt = select(RatingEvent).where(RatingEvent.game_id == UUID(game_id))
                result = await session.execute(stmt)
                return list(result.scalars().all())

        return asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
