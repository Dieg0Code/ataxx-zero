from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.config import Settings


class TestApiSettingsSecurity(unittest.TestCase):
    _short_secret = "short-secret"  # noqa: S105
    _strong_secret = "this-is-a-very-strong-jwt-secret-12345"  # noqa: S105

    def test_non_production_allows_empty_jwt_secret(self) -> None:
        settings = Settings(app_env="development", auth_jwt_secret="")
        self.assertEqual(settings.app_env, "development")

    def test_production_rejects_empty_jwt_secret(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(app_env="production", auth_jwt_secret="")

    def test_production_rejects_short_jwt_secret(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(app_env="prod", auth_jwt_secret=self._short_secret)

    def test_production_accepts_strong_jwt_secret(self) -> None:
        settings = Settings(
            app_env="production",
            auth_jwt_secret=self._strong_secret,
        )
        self.assertEqual(settings.app_env, "production")


if __name__ == "__main__":
    unittest.main()
