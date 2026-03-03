from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.db.enums import AgentType, BotKind, GameStatus
from api.db.models import BotProfile, Game, ModelVersion, User
from api.modules.matches.repository import MatchesRepository
from api.modules.matches.service import MatchesService
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from inference.service import InferenceResult, InferenceService


class _StubInferenceService:
    def predict(self, board: AtaxxBoard, mode: str = "fast") -> InferenceResult:
        legal = board.get_valid_moves()
        move = legal[0] if legal else None
        action_idx = ACTION_SPACE.pass_index if move is None else ACTION_SPACE.encode(move)
        resolved_mode = "strong" if mode == "strong" else "fast"
        return InferenceResult(move=move, action_idx=action_idx, value=0.0, mode=resolved_mode)


class TestMatchesServiceModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "matches_service_model_inference.db"
        cls.engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        cls.sessionmaker = async_sessionmaker(
            bind=cls.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async def _init_db() -> None:
            async with cls.engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)

        asyncio.run(_init_db())

    @classmethod
    def tearDownClass(cls) -> None:
        async def _dispose() -> None:
            await cls.engine.dispose()

        asyncio.run(_dispose())
        cls.tmpdir.cleanup()

    def test_model_bot_resolves_account_model_version_runtime(self) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                service = MatchesService(repository=MatchesRepository(session=session))
                human = User(username="human-a", email="human-a@example.com", is_active=True)
                version = ModelVersion(
                    name="ub_bogonet_v0",
                    checkpoint_uri="checkpoints/bogonet_v0.ckpt",
                    is_active=False,
                )
                bot = User(
                    username="ub_bogonet_v0",
                    email="bogo@example.com",
                    is_active=True,
                    is_bot=True,
                    bot_kind=BotKind.MODEL,
                    model_version_id=version.id,
                )
                session.add(human)
                session.add(version)
                session.add(bot)
                session.add(
                    BotProfile(
                        user_id=bot.id,
                        agent_type=AgentType.MODEL,
                        model_mode="fast",
                        enabled=True,
                    )
                )
                game = Game(
                    status=GameStatus.IN_PROGRESS,
                    player1_id=bot.id,
                    player2_id=human.id,
                    created_by_user_id=human.id,
                    player1_agent=AgentType.MODEL,
                    player2_agent=AgentType.HUMAN,
                    model_version_id=version.id,
                )
                session.add(game)
                await session.commit()

                human_db = await session.get(User, human.id)
                if human_db is None:
                    self.fail("Human user not found")

                with patch(
                    "api.modules.matches.service.resolve_model_inference_service",
                    return_value=_StubInferenceService(),
                ) as resolver:
                    move = await service.advance_bot_turn(
                        game_id=game.id,
                        actor_user=human_db,
                        inference_service=None,
                    )
                self.assertIsNotNone(move)
                if move is None:
                    self.fail("Expected a persisted bot move")
                self.assertEqual(move.mode, "fast")
                resolver.assert_called_once()

        asyncio.run(_run())

    def test_model_bot_falls_back_to_global_inference_when_version_has_no_artifacts(self) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                service = MatchesService(repository=MatchesRepository(session=session))
                human = User(username="human-b", email="human-b@example.com", is_active=True)
                version = ModelVersion(name="ub_empty_model", checkpoint_uri=None, onnx_uri=None, is_active=False)
                bot = User(
                    username="ub_model_fallback",
                    email="fallback@example.com",
                    is_active=True,
                    is_bot=True,
                    bot_kind=BotKind.MODEL,
                    model_version_id=version.id,
                )
                session.add(human)
                session.add(version)
                session.add(bot)
                session.add(
                    BotProfile(
                        user_id=bot.id,
                        agent_type=AgentType.MODEL,
                        model_mode="strong",
                        enabled=True,
                    )
                )
                game = Game(
                    status=GameStatus.IN_PROGRESS,
                    player1_id=bot.id,
                    player2_id=human.id,
                    created_by_user_id=human.id,
                    player1_agent=AgentType.MODEL,
                    player2_agent=AgentType.HUMAN,
                    model_version_id=version.id,
                )
                session.add(game)
                await session.commit()

                human_db = await session.get(User, human.id)
                if human_db is None:
                    self.fail("Human user not found")

                move = await service.advance_bot_turn(
                    game_id=game.id,
                    actor_user=human_db,
                    inference_service=cast(InferenceService, _StubInferenceService()),
                )
                self.assertIsNotNone(move)
                if move is None:
                    self.fail("Expected a persisted bot move")
                self.assertEqual(move.mode, "strong")

        asyncio.run(_run())

    def test_create_invitation_prewarms_runtime_for_model_bot(self) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                service = MatchesService(repository=MatchesRepository(session=session))
                human = User(username="human-c", email="human-c@example.com", is_active=True)
                version = ModelVersion(
                    name="ub_policy_spatial_v2",
                    checkpoint_uri="checkpoints/policy_spatial_v2.ckpt",
                    is_active=False,
                )
                bot = User(
                    username="ub_bogonet_warmup",
                    email="bogonet@example.com",
                    is_active=True,
                    is_bot=True,
                    bot_kind=BotKind.MODEL,
                    model_version_id=version.id,
                )
                session.add(human)
                session.add(version)
                session.add(bot)
                session.add(
                    BotProfile(
                        user_id=bot.id,
                        agent_type=AgentType.MODEL,
                        model_mode="fast",
                        enabled=True,
                    )
                )
                await session.commit()

                with patch("api.modules.matches.service.prewarm_model_inference_service") as prewarm:
                    game = await service.create_invitation(
                        actor_user_id=human.id,
                        opponent_user_id=bot.id,
                        rated=False,
                    )

                self.assertEqual(game.status, GameStatus.IN_PROGRESS)
                self.assertEqual(game.player2_agent, AgentType.MODEL)
                prewarm.assert_called_once()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
