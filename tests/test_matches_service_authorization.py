from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.db.enums import GameStatus
from api.db.models import Game, User
from api.modules.matches.repository import MatchesRepository
from api.modules.matches.service import MatchesService


class TestMatchesServiceAuthorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "matches_service_auth.db"
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

    def test_admin_can_view_any_match(self) -> None:
        async def _run() -> None:
            async with self.sessionmaker() as session:
                repo = MatchesRepository(session=session)
                service = MatchesService(repository=repo, ranking_service=None)

                p1_id = uuid4()
                p2_id = uuid4()
                outsider_id = uuid4()

                game = await repo.create_game(
                    Game(
                        status=GameStatus.IN_PROGRESS,
                        player1_id=p1_id,
                        player2_id=p2_id,
                    )
                )

                admin_user = User(
                    id=uuid4(),
                    username="admin-user",
                    email="admin@example.com",
                    password_hash=None,
                    is_active=True,
                    is_admin=True,
                    is_bot=False,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                p1_user = User(
                    id=p1_id,
                    username="p1-user",
                    email="p1@example.com",
                    password_hash=None,
                    is_active=True,
                    is_admin=False,
                    is_bot=False,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                outsider_user = User(
                    id=outsider_id,
                    username="outsider-user",
                    email="outsider@example.com",
                    password_hash=None,
                    is_active=True,
                    is_admin=False,
                    is_bot=False,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                await service.ensure_can_view_match(
                    game_id=game.id,
                    actor_user=admin_user,
                )
                await service.ensure_can_view_match(
                    game_id=game.id,
                    actor_user=p1_user,
                )
                with self.assertRaises(PermissionError):
                    await service.ensure_can_view_match(
                        game_id=game.id,
                        actor_user=outsider_user,
                    )

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
