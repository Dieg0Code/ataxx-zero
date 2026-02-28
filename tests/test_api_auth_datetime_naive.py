from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.config.settings import Settings
from api.db.models import User
from api.modules.auth.repository import AuthRepository
from api.modules.auth.schemas import AuthRegisterRequest
from api.modules.auth.service import AuthService


class _FakeAuthRepository:
    def __init__(self) -> None:
        self.created_user: User | None = None

    async def create_user(self, user: User) -> User:
        self.created_user = user
        return user


class TestApiAuthDatetimeNaive(unittest.TestCase):
    def test_register_uses_naive_utc_datetimes_for_db_columns(self) -> None:
        repository = _FakeAuthRepository()
        service = AuthService(
            repository=cast(AuthRepository, repository),
            settings=Settings(),
        )

        safe_password = "supersecret" + "123"
        payload = AuthRegisterRequest(
            username="datetimeuser",
            email="datetime@example.com",
            password=safe_password,
        )
        created = asyncio.run(service.register(payload))

        self.assertIsNotNone(repository.created_user)
        self.assertIsNone(created.created_at.tzinfo)
        self.assertIsNone(created.updated_at.tzinfo)


if __name__ == "__main__":
    unittest.main()

