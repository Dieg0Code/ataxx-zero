from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from inference.checkpoint_league_runtime import (
    DEFAULT_RATING,
    apply_series_to_league,
    choose_champion_id,
    compute_expected_score,
    create_empty_league_state,
    load_league_state,
    save_league_state,
)


class TestInferenceCheckpointLeagueRuntime(unittest.TestCase):
    def test_compute_expected_score_is_symmetric(self) -> None:
        expected_a = compute_expected_score(rating_a=1200.0, rating_b=1400.0)
        expected_b = compute_expected_score(rating_a=1400.0, rating_b=1200.0)

        self.assertAlmostEqual(expected_a + expected_b, 1.0)
        self.assertLess(expected_a, 0.5)
        self.assertGreater(expected_b, 0.5)

    def test_apply_series_to_league_initializes_entries_and_updates_champion(self) -> None:
        league = create_empty_league_state()

        summary = apply_series_to_league(
            league=league,
            participant_a_id="ckpt:v2_093",
            participant_a_name="policy_spatial_v2_iter_093",
            participant_b_id="ckpt:v5_010",
            participant_b_name="policy_spatial_v5_iter_010",
            participant_a_artifact_path="checkpoints/policy_spatial_v2_iter_093.pt",
            participant_b_artifact_path="checkpoints/policy_spatial_v5_iter_010.pt",
            series_summary={
                "games": 8,
                "checkpoint_a_wins": 6,
                "checkpoint_b_wins": 2,
                "draws": 0,
                "checkpoint_a_score": 0.75,
                "avg_turns": 40.0,
            },
        )

        self.assertEqual(summary["games"], 8)
        self.assertEqual(league["champion_id"], "ckpt:v2_093")
        self.assertIn("ckpt:v2_093", league["entries"])
        self.assertIn("ckpt:v5_010", league["entries"])
        self.assertEqual(
            league["entries"]["ckpt:v2_093"]["artifact_path"],
            "checkpoints/policy_spatial_v2_iter_093.pt",
        )
        self.assertGreater(
            float(league["entries"]["ckpt:v2_093"]["rating"]),
            DEFAULT_RATING,
        )
        self.assertLess(
            float(league["entries"]["ckpt:v5_010"]["rating"]),
            DEFAULT_RATING,
        )

    def test_choose_champion_id_prefers_higher_rating_then_games(self) -> None:
        league = create_empty_league_state()
        league["entries"] = {
            "ckpt:a": {
                "participant_id": "ckpt:a",
                "display_name": "A",
                "kind": "checkpoint",
                "rating": 1224.0,
                "games_played": 4,
                "wins": 3,
                "losses": 1,
                "draws": 0,
            },
            "ckpt:b": {
                "participant_id": "ckpt:b",
                "display_name": "B",
                "kind": "checkpoint",
                "rating": 1224.0,
                "games_played": 8,
                "wins": 5,
                "losses": 3,
                "draws": 0,
            },
        }

        self.assertEqual(choose_champion_id(league), "ckpt:b")

    def test_choose_champion_id_ignores_heuristics_when_checkpoint_exists(self) -> None:
        league = create_empty_league_state()
        league["entries"] = {
            "heu:sentinel": {
                "participant_id": "heu:sentinel",
                "display_name": "sentinel",
                "kind": "heuristic",
                "rating": 1400.0,
                "games_played": 20,
                "wins": 15,
                "losses": 5,
                "draws": 0,
            },
            "ckpt:v2": {
                "participant_id": "ckpt:v2",
                "display_name": "v2",
                "kind": "checkpoint",
                "rating": 1250.0,
                "games_played": 10,
                "wins": 7,
                "losses": 3,
                "draws": 0,
            },
        }

        self.assertEqual(choose_champion_id(league), "ckpt:v2")

    def test_save_and_load_league_state_roundtrip(self) -> None:
        league = create_empty_league_state()
        apply_series_to_league(
            league=league,
            participant_a_id="ckpt:v2_093",
            participant_a_name="policy_spatial_v2_iter_093",
            participant_b_id="heu:sentinel",
            participant_b_name="sentinel",
            series_summary={
                "games": 4,
                "checkpoint_a_wins": 1,
                "checkpoint_b_wins": 1,
                "draws": 2,
                "checkpoint_a_score": 0.5,
                "avg_turns": 32.0,
            },
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "league.json"
            save_league_state(path=path, league=league)
            loaded = load_league_state(path=path)

        self.assertEqual(loaded["champion_id"], league["champion_id"])
        self.assertEqual(loaded["entries"].keys(), league["entries"].keys())
        self.assertEqual(
            json.dumps(loaded, sort_keys=True),
            json.dumps(league, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
