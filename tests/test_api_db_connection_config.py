from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.config import Settings
from api.db import session as db_session


class TestApiDbConnectionConfig(unittest.TestCase):
    def test_sqlalchemy_database_url_for_asyncpg_has_no_options_query(self) -> None:
        settings = Settings(
            database_url="",
            db_host="localhost",
            db_port=5432,
            db_name="ataxx_zero",
            db_user="postgres",
            db_password="secret",  # noqa: S106
            db_timezone="UTC",
        )

        url = settings.sqlalchemy_database_url

        self.assertTrue(url.startswith("postgresql+asyncpg://"))
        self.assertNotIn("?options=", url)
        self.assertNotIn("&options=", url)

    def test_database_url_override_is_used_verbatim(self) -> None:
        direct = "postgresql+asyncpg://u:p@db:5432/name"
        settings = Settings(database_url=direct)
        self.assertEqual(settings.sqlalchemy_database_url, direct)

    def test_engine_uses_server_settings_timezone_for_asyncpg(self) -> None:
        db_session.get_engine.cache_clear()
        db_session.get_settings.cache_clear()

        with patch.object(
            db_session,
            "get_settings",
            return_value=Settings(
                database_url="postgresql+asyncpg://u:p@db:5432/name",
                db_timezone="America/Santiago",
                db_pool_size=5,
                db_max_overflow=7,
                db_pool_timeout_s=11,
                db_pool_recycle_s=13,
            ),
        ), patch.object(db_session, "create_async_engine") as create_engine_mock:
            db_session.get_engine()

        self.assertTrue(create_engine_mock.called)
        _, kwargs = create_engine_mock.call_args
        self.assertEqual(
            kwargs.get("connect_args"),
            {"server_settings": {"timezone": "America/Santiago"}},
        )


if __name__ == "__main__":
    unittest.main()
