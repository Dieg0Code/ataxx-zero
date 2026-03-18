from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from training.league_runtime import (
    build_checkpoint_pool_specs,
    load_checkpoint_pool_entries,
    record_checkpoint_in_league,
    resolve_champion_entry,
)


def _write_checkpoint(path: Path) -> None:
    payload = {
        "state_dict": {
            "model.linear.weight": torch.zeros((1, 1), dtype=torch.float32),
            "model.linear.bias": torch.zeros((1,), dtype=torch.float32),
        },
        "hparams": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
    }
    torch.save(payload, path)


class TestTrainingLeagueRuntime(unittest.TestCase):
    def test_load_checkpoint_pool_entries_filters_and_sorts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            keep_path = tmp_path / "keep.ckpt"
            current_path = tmp_path / "current.ckpt"
            _write_checkpoint(keep_path)
            _write_checkpoint(current_path)
            league_path = tmp_path / "league.json"
            league_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "ckpt:keep": {
                                "participant_id": "ckpt:keep",
                                "display_name": "keep",
                                "kind": "checkpoint",
                                "rating": 1250.0,
                                "games_played": 8,
                                "artifact_path": str(keep_path),
                            },
                            "ckpt:current": {
                                "participant_id": "ckpt:current",
                                "display_name": "current",
                                "kind": "checkpoint",
                                "rating": 1400.0,
                                "games_played": 8,
                                "artifact_path": str(current_path),
                            },
                            "heu:sentinel": {
                                "participant_id": "heu:sentinel",
                                "display_name": "sentinel",
                                "kind": "heuristic",
                                "rating": 1500.0,
                                "games_played": 10,
                            },
                        },
                    },
                ),
                encoding="utf-8",
            )

            with patch("training.league_runtime.cfg_bool", side_effect=lambda key: key == "league_enabled"), patch(
                "training.league_runtime.cfg_str",
                side_effect=lambda key: str(league_path) if key == "league_path" else "",
            ), patch(
                "training.league_runtime.cfg_int",
                side_effect=lambda key: {"league_checkpoint_pool_min_games": 4, "league_checkpoint_pool_top_k": 3}[key],
            ):
                entries = load_checkpoint_pool_entries(current_checkpoint_path=current_path)

        self.assertEqual([entry.participant_id for entry in entries], ["ckpt:keep"])

    def test_build_checkpoint_pool_specs_loads_state_dict(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            keep_path = tmp_path / "keep.ckpt"
            _write_checkpoint(keep_path)
            league_path = tmp_path / "league.json"
            league_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "ckpt:keep": {
                                "participant_id": "ckpt:keep",
                                "display_name": "keep",
                                "kind": "checkpoint",
                                "rating": 1250.0,
                                "games_played": 8,
                                "artifact_path": str(keep_path),
                            },
                        },
                    },
                ),
                encoding="utf-8",
            )
            with patch("training.league_runtime.cfg_bool", side_effect=lambda key: key == "league_enabled"), patch(
                "training.league_runtime.cfg_str",
                side_effect=lambda key: str(league_path) if key == "league_path" else "",
            ), patch(
                "training.league_runtime.cfg_int",
                side_effect=lambda key: {"league_checkpoint_pool_min_games": 0, "league_checkpoint_pool_top_k": 1}[key],
            ):
                specs = build_checkpoint_pool_specs(current_checkpoint_path=None)

        self.assertIn("ckpt:keep", specs)
        self.assertIn("linear.weight", specs["ckpt:keep"].state_dict)

    def test_record_checkpoint_in_league_persists_artifact_and_champion(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "manual_iter_012.ckpt"
            champion_path = tmp_path / "manual_iter_006.ckpt"
            _write_checkpoint(checkpoint_path)
            _write_checkpoint(champion_path)
            league_path = tmp_path / "league.json"

            with patch("training.league_runtime.cfg_str", side_effect=lambda key: str(league_path) if key == "league_path" else ""), patch(
                "training.league_runtime.cfg_bool",
                side_effect=lambda key: key == "league_enabled",
            ):
                league = record_checkpoint_in_league(
                    checkpoint_path=checkpoint_path,
                    heuristic_series_by_level={
                        "hard": {
                            "games": 4,
                            "checkpoint_a_wins": 2,
                            "checkpoint_b_wins": 1,
                            "draws": 1,
                            "checkpoint_a_score": 0.625,
                            "avg_turns": 30.0,
                        },
                    },
                    champion_entry=None,
                    champion_series_summary=None,
                )

        self.assertEqual(
            league["entries"]["ckpt:manual_iter_012"]["artifact_path"],
            str(checkpoint_path),
        )
        self.assertEqual(league["champion_id"], "ckpt:manual_iter_012")

    def test_record_checkpoint_in_league_accepts_eval_style_heuristic_summary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "manual_iter_018.ckpt"
            _write_checkpoint(checkpoint_path)
            league_path = tmp_path / "league.json"

            with patch("training.league_runtime.cfg_str", side_effect=lambda key: str(league_path) if key == "league_path" else ""), patch(
                "training.league_runtime.cfg_bool",
                side_effect=lambda key: key == "league_enabled",
            ):
                league = record_checkpoint_in_league(
                    checkpoint_path=checkpoint_path,
                    heuristic_series_by_level={
                        "apex": {
                            "games": 6,
                            "wins": 4,
                            "losses": 1,
                            "draws": 1,
                            "score": 0.75,
                        },
                    },
                    champion_entry=None,
                    champion_series_summary=None,
                )

        recent_series = league["recent_series"]
        self.assertEqual(len(recent_series), 1)
        self.assertEqual(recent_series[0]["checkpoint_a_wins"], 4)
        self.assertEqual(recent_series[0]["checkpoint_b_wins"], 1)
        self.assertEqual(recent_series[0]["draws"], 1)
        self.assertEqual(recent_series[0]["avg_turns"], 0.0)

    def test_resolve_champion_entry_returns_checkpoint_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            champion_path = tmp_path / "champ.ckpt"
            _write_checkpoint(champion_path)
            league_path = tmp_path / "league.json"
            league_path.write_text(
                json.dumps(
                    {
                        "champion_id": "ckpt:champ",
                        "entries": {
                            "heu:sentinel": {
                                "participant_id": "heu:sentinel",
                                "display_name": "sentinel",
                                "kind": "heuristic",
                                "rating": 1400.0,
                                "games_played": 20,
                            },
                            "ckpt:champ": {
                                "participant_id": "ckpt:champ",
                                "display_name": "champ",
                                "kind": "checkpoint",
                                "rating": 1300.0,
                                "games_played": 10,
                                "artifact_path": str(champion_path),
                            },
                        },
                    },
                ),
                encoding="utf-8",
            )

            with patch("training.league_runtime.cfg_bool", side_effect=lambda key: key == "league_enabled"), patch(
                "training.league_runtime.cfg_str",
                side_effect=lambda key: str(league_path) if key == "league_path" else "",
            ):
                champion = resolve_champion_entry(current_checkpoint_path=None)

        self.assertIsNotNone(champion)
        self.assertEqual(champion.participant_id, "ckpt:champ")


if __name__ == "__main__":
    unittest.main()
