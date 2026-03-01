from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.db import models as _models
from api.db.enums import AgentType, BotKind, PlayerSide
from api.db.models import BotProfile, GameMove, Season, User
from api.db.session import get_session

del _models


class TestApiMatchesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "matches_integration.db"
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

    def test_create_match_and_get_initial_state(self) -> None:
        p1 = self._register_and_login("match-p1", "match-p1@example.com")
        p2 = self._register_and_login("match-p2", "match-p2@example.com")
        outsider = self._register_and_login("match-outside", "match-outside@example.com")

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "queue_type": "casual",
                "player1_id": p1["user_id"],
                "player2_id": p2["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        get_resp = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(get_resp.status_code, 200)

        state_resp = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(state_resp.status_code, 200)
        state = state_resp.json()
        self.assertEqual(state["game_id"], game_id)
        self.assertEqual(state["status"], "in_progress")
        self.assertEqual(state["board"]["current_player"], 1)
        self.assertEqual(state["next_player_side"], "p1")
        self.assertGreater(len(state["legal_moves"]), 0)

        outsider_get = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_get.status_code, 403)

        outsider_state = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_state.status_code, 403)

    def test_submit_move_validates_turn_and_legality(self) -> None:
        p1 = self._register_and_login("match-p1b", "match-p1b@example.com")
        p2 = self._register_and_login("match-p2b", "match-p2b@example.com")
        outsider = self._register_and_login("match-out", "match-out@example.com")

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "player1_id": p1["user_id"],
                "player2_id": p2["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        spoof_resp = self.client.post(
            "/api/v1/matches",
            json={
                "player1_id": p2["user_id"],
                "player2_id": p1["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(spoof_resp.status_code, 403)

        first_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 0, "c1": 0, "r2": 1, "c2": 1},
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(first_move.status_code, 201)
        first = first_move.json()
        self.assertEqual(first["ply"], 0)
        self.assertEqual(first["player_side"], "p1")

        outsider_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 6, "c1": 0, "r2": 5, "c2": 1},
            },
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_move.status_code, 403)

        wrong_turn = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 0, "c1": 6, "r2": 1, "c2": 5},
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(wrong_turn.status_code, 403)

        illegal_move = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={
                "pass_turn": False,
                "move": {"r1": 6, "c1": 0, "r2": 6, "c2": 0},
            },
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(illegal_move.status_code, 400)

        illegal_pass = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={"pass_turn": True},
            headers={"Authorization": f"Bearer {p2['access_token']}"},
        )
        self.assertEqual(illegal_pass.status_code, 400)

    def test_admin_can_read_match_http_end_to_end(self) -> None:
        p1 = self._register_and_login("match-ap1", "match-ap1@example.com")
        p2 = self._register_and_login("match-ap2", "match-ap2@example.com")
        outsider = self._register_and_login("match-aout", "match-aout@example.com")
        admin = self._register_and_login("match-admin", "match-admin@example.com")

        self._promote_user_to_admin(admin["user_id"])

        create_resp = self.client.post(
            "/api/v1/matches",
            json={
                "player1_id": p1["user_id"],
                "player2_id": p2["user_id"],
            },
            headers={"Authorization": f"Bearer {p1['access_token']}"},
        )
        self.assertEqual(create_resp.status_code, 201)
        game_id = create_resp.json()["id"]

        outsider_get = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_get.status_code, 403)

        outsider_state = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_state.status_code, 403)

        admin_get = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(admin_get.status_code, 200)
        self.assertEqual(admin_get.json()["id"], game_id)

        admin_state = self.client.get(
            f"/api/v1/matches/{game_id}/state",
            headers={"Authorization": f"Bearer {admin['access_token']}"},
        )
        self.assertEqual(admin_state.status_code, 200)
        self.assertEqual(admin_state.json()["game_id"], game_id)

    def test_human_invitation_flow_create_list_accept_reject(self) -> None:
        self._create_active_season()
        inviter = self._register_and_login("invite-p1", "invite-p1@example.com")
        invited = self._register_and_login("invite-p2", "invite-p2@example.com")
        outsider = self._register_and_login("invite-out", "invite-out@example.com")

        create_invite = self.client.post(
            "/api/v1/matches/invitations",
            json={"opponent_user_id": invited["user_id"]},
            headers={"Authorization": f"Bearer {inviter['access_token']}"},
        )
        self.assertEqual(create_invite.status_code, 201)
        invitation_id = create_invite.json()["id"]
        self.assertEqual(create_invite.json()["status"], "pending")
        self.assertEqual(create_invite.json()["queue_type"], "custom")

        incoming = self.client.get(
            "/api/v1/matches/invitations/incoming?limit=10&offset=0",
            headers={"Authorization": f"Bearer {invited['access_token']}"},
        )
        self.assertEqual(incoming.status_code, 200)
        items = incoming.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], invitation_id)

        outsider_accept = self.client.post(
            f"/api/v1/matches/invitations/{invitation_id}/accept",
            headers={"Authorization": f"Bearer {outsider['access_token']}"},
        )
        self.assertEqual(outsider_accept.status_code, 403)

        invited_accept = self.client.post(
            f"/api/v1/matches/invitations/{invitation_id}/accept",
            headers={"Authorization": f"Bearer {invited['access_token']}"},
        )
        self.assertEqual(invited_accept.status_code, 200)
        self.assertEqual(invited_accept.json()["status"], "in_progress")

        create_invite_2 = self.client.post(
            "/api/v1/matches/invitations",
            json={"opponent_user_id": invited["user_id"]},
            headers={"Authorization": f"Bearer {inviter['access_token']}"},
        )
        self.assertEqual(create_invite_2.status_code, 201)
        invitation_id_2 = create_invite_2.json()["id"]

        invited_reject = self.client.post(
            f"/api/v1/matches/invitations/{invitation_id_2}/reject",
            headers={"Authorization": f"Bearer {invited['access_token']}"},
        )
        self.assertEqual(invited_reject.status_code, 200)
        self.assertEqual(invited_reject.json()["status"], "aborted")

    def test_bot_invitation_auto_accepts(self) -> None:
        self._create_active_season()
        inviter = self._register_and_login("invite-bot-p1", "invite-bot-p1@example.com")
        bot = self._register_and_login("invite-bot-p2", "invite-bot-p2@example.com")
        self._mark_user_as_heuristic_bot(bot["user_id"])

        create_invite = self.client.post(
            "/api/v1/matches/invitations",
            json={"opponent_user_id": bot["user_id"]},
            headers={"Authorization": f"Bearer {inviter['access_token']}"},
        )
        self.assertEqual(create_invite.status_code, 201)
        payload = create_invite.json()
        self.assertEqual(payload["status"], "in_progress")
        self.assertEqual(payload["queue_type"], "custom")
        self.assertEqual(payload["player1_agent"], "human")
        self.assertEqual(payload["player2_agent"], "heuristic")

    def test_human_invitation_ranked_applies_rating(self) -> None:
        season_id = self._create_active_season()
        inviter = self._register_and_login("invite-rated-p1", "invite-rated-p1@example.com")
        invited = self._register_and_login("invite-rated-p2", "invite-rated-p2@example.com")

        invite_resp = self.client.post(
            "/api/v1/matches/invitations",
            json={"opponent_user_id": invited["user_id"]},
            headers={"Authorization": f"Bearer {inviter['access_token']}"},
        )
        self.assertEqual(invite_resp.status_code, 201)
        invitation = invite_resp.json()
        self.assertTrue(invitation["rated"])
        self.assertEqual(invitation["season_id"], season_id)

        game_id = invitation["id"]
        accept_resp = self.client.post(
            f"/api/v1/matches/invitations/{game_id}/accept",
            headers={"Authorization": f"Bearer {invited['access_token']}"},
        )
        self.assertEqual(accept_resp.status_code, 200)
        self.assertEqual(accept_resp.json()["status"], "in_progress")

        self._seed_near_terminal_human_match(game_id=game_id)
        finish_resp = self.client.post(
            f"/api/v1/matches/{game_id}/moves",
            json={"pass_turn": False, "move": {"r1": 0, "c1": 0, "r2": 0, "c2": 1}},
            headers={"Authorization": f"Bearer {inviter['access_token']}"},
        )
        self.assertEqual(finish_resp.status_code, 201)

        game_resp = self.client.get(
            f"/api/v1/matches/{game_id}",
            headers={"Authorization": f"Bearer {inviter['access_token']}"},
        )
        self.assertEqual(game_resp.status_code, 200)
        self.assertEqual(game_resp.json()["status"], "finished")
        self.assertEqual(game_resp.json()["winner_side"], "p1")

        p1_rating_resp = self.client.get(f"/api/v1/ranking/ratings/{inviter['user_id']}/{season_id}")
        p2_rating_resp = self.client.get(f"/api/v1/ranking/ratings/{invited['user_id']}/{season_id}")
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

    def test_invitations_ws_streams_new_invitation(self) -> None:
        inviter = self._register_and_login("invite-ws-p1", "invite-ws-p1@example.com")
        invited = self._register_and_login("invite-ws-p2", "invite-ws-p2@example.com")

        with self.client.websocket_connect(
            f"/api/v1/matches/invitations/ws?token={invited['access_token']}"
        ) as websocket:
            subscribed = websocket.receive_json()
            self.assertEqual(subscribed["type"], "invitations.subscribed")

            initial = websocket.receive_json()
            self.assertEqual(initial["type"], "invitations.status")
            self.assertEqual(initial["payload"]["total"], 0)

            create_invite = self.client.post(
                "/api/v1/matches/invitations",
                json={"opponent_user_id": invited["user_id"]},
                headers={"Authorization": f"Bearer {inviter['access_token']}"},
            )
            self.assertEqual(create_invite.status_code, 201)
            invitation_id = create_invite.json()["id"]

            for _ in range(6):
                event = websocket.receive_json()
                if event.get("type") != "invitations.status":
                    continue
                items = event["payload"]["items"]
                if event["payload"]["total"] > 0 and any(row["id"] == invitation_id for row in items):
                    break
            else:
                self.fail("Invitation not streamed over websocket within timeout window.")

    def test_invitations_ws_rejects_missing_token(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as ctx, self.client.websocket_connect(
            "/api/v1/matches/invitations/ws"
        ) as websocket:
            websocket.receive_json()
        self.assertEqual(ctx.exception.code, 4401)

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

    def _mark_user_as_heuristic_bot(self, user_id: str) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                user = await session.get(User, UUID(user_id))
                if user is None:
                    raise AssertionError("User not found for bot promotion")
                user.is_bot = True
                user.bot_kind = BotKind.HEURISTIC
                user.is_hidden_bot = False
                session.add(user)
                session.add(
                    BotProfile(
                        user_id=user.id,
                        agent_type=AgentType.HEURISTIC,
                        heuristic_level="normal",
                        model_mode=None,
                        enabled=True,
                    )
                )
                await session.commit()

        asyncio.run(_run())

    def _create_active_season(self) -> str:
        async def _run() -> str:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            async with self.sessionmaker() as session:
                rows = await session.execute(select(Season))
                for season in rows.scalars().all():
                    season.is_active = False
                    session.add(season)
                season = Season(
                    name=f"Season Match {now.timestamp()}",
                    starts_at=now - timedelta(days=1),
                    ends_at=now + timedelta(days=30),
                    is_active=True,
                )
                session.add(season)
                await session.commit()
                await session.refresh(season)
                return str(season.id)

        return asyncio.run(_run())

    def _seed_near_terminal_human_match(self, *, game_id: str) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                seed_grid = [[0 for _ in range(7)] for _ in range(7)]
                seed_grid[0][0] = 1
                seed_grid[0][2] = -1
                seed_state = {
                    "grid": seed_grid,
                    "current_player": 1,
                    "half_moves": 0,
                }
                move = GameMove(
                    game_id=UUID(game_id),
                    ply=0,
                    player_side=PlayerSide.P2,
                    r1=None,
                    c1=None,
                    r2=None,
                    c2=None,
                    board_before=seed_state,
                    board_after=seed_state,
                    mode="seed",
                    action_idx=0,
                    value=0.0,
                )
                session.add(move)
                await session.commit()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
