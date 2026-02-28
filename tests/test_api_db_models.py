from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sqlmodel import SQLModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.db import models as _models

del _models


class TestApiDbModels(unittest.TestCase):
    def test_expected_tables_are_registered(self) -> None:
        table_names = set(SQLModel.metadata.tables.keys())
        self.assertIn("user", table_names)
        self.assertIn("botprofile", table_names)
        self.assertIn("game", table_names)
        self.assertIn("gamemove", table_names)
        self.assertIn("leaderboardentry", table_names)
        self.assertIn("modelversion", table_names)
        self.assertIn("season", table_names)
        self.assertIn("rating", table_names)
        self.assertIn("ratingevent", table_names)
        self.assertIn("trainingsample", table_names)
        self.assertIn("authrefreshtoken", table_names)

    def test_users_has_bot_columns(self) -> None:
        users_table = SQLModel.metadata.tables["user"]
        self.assertIn("is_bot", users_table.c)
        self.assertIn("bot_kind", users_table.c)
        self.assertIn("is_hidden_bot", users_table.c)
        self.assertIn("model_version_id", users_table.c)

    def test_bot_profile_columns(self) -> None:
        bot_table = SQLModel.metadata.tables["botprofile"]
        self.assertIn("user_id", bot_table.c)
        self.assertIn("agent_type", bot_table.c)
        self.assertIn("heuristic_level", bot_table.c)
        self.assertIn("model_mode", bot_table.c)
        self.assertIn("enabled", bot_table.c)

    def test_games_has_training_columns(self) -> None:
        games_table = SQLModel.metadata.tables["game"]
        self.assertIn("source", games_table.c)
        self.assertIn("quality_score", games_table.c)
        self.assertIn("is_training_eligible", games_table.c)

    def test_game_moves_has_replay_columns(self) -> None:
        moves_table = SQLModel.metadata.tables["gamemove"]
        self.assertIn("board_before", moves_table.c)
        self.assertIn("board_after", moves_table.c)

    def test_model_version_unique_name_constraint_exists(self) -> None:
        constraints = SQLModel.metadata.tables["modelversion"].constraints
        has_unique_name = any(
            getattr(constraint, "name", "") == "uq_model_versions_name"
            for constraint in constraints
        )
        self.assertTrue(has_unique_name)

    def test_training_samples_has_targets_columns(self) -> None:
        samples_table = SQLModel.metadata.tables["trainingsample"]
        self.assertIn("observation", samples_table.c)
        self.assertIn("policy_target", samples_table.c)
        self.assertIn("value_target", samples_table.c)


if __name__ == "__main__":
    unittest.main()
