from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import cast
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.modules.gameplay.repository import GameRepository


class _CountOnlyResult:
    def scalar_one(self) -> int:
        return 7


class _CountOnlySession:
    async def execute(self, _stmt: object) -> _CountOnlyResult:
        return _CountOnlyResult()


class TestGameRepositoryUnit(unittest.TestCase):
    def test_next_ply_uses_count_result(self) -> None:
        async def _run() -> None:
            repo = GameRepository(cast(AsyncSession, _CountOnlySession()))
            ply = await repo.next_ply(uuid4())
            self.assertEqual(ply, 7)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
