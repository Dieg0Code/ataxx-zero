from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import torch

from training.monitor import TrainingMonitor


class TestTrainingMonitor(unittest.TestCase):
    def test_log_iteration_base_and_extended_lines(self) -> None:
        monitor = TrainingMonitor(total_iterations=10, log_every=2)
        selfplay_stats = {
            "cache_hit_rate": 0.33,
            "wins_p1": 4,
            "wins_p2": 3,
            "draws": 1,
            "avg_game_length": 52.0,
            "episodes_vs_self": 2,
            "episodes_vs_heuristic": 5,
            "episodes_vs_random": 1,
            "episodes_vs_heuristic_easy": 1,
            "episodes_vs_heuristic_normal": 2,
            "episodes_vs_heuristic_hard": 2,
            "episodes_vs_heuristic_apex": 1,
            "episodes_vs_heuristic_gambit": 0,
            "episodes_vs_heuristic_sentinel": 0,
        }
        metrics = {
            "train/loss": torch.tensor(2.0),
            "train/loss_value": torch.tensor(0.4),
            "train/policy_accuracy": torch.tensor(0.25),
            "val/loss": torch.tensor(1.8),
        }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            monitor.log_iteration(
                iteration=2,
                selfplay_s=38.0,
                fit_s=9.0,
                buffer_size=3200,
                selfplay_stats=selfplay_stats,
                logged_metrics=metrics,
            )
        text = buffer.getvalue()
        self.assertIn("[02/10] sp=38s tr=9s buf=3200", text)
        self.assertIn("W/L/D=4/3/1", text)
        self.assertIn("heuristic easy:17% normal:33% hard:33% apex:17% gambit:0% sentinel:0%", text)

    def test_log_iteration_alerts(self) -> None:
        monitor = TrainingMonitor(total_iterations=10, log_every=5)
        selfplay_stats = {"cache_hit_rate": 0.1, "avg_game_length": 90.0}
        metrics = {
            "train/loss": torch.tensor(1.0),
            "train/loss_value": torch.tensor(0.9),
        }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            monitor.log_iteration(
                iteration=6,
                selfplay_s=40.0,
                fit_s=8.0,
                buffer_size=5000,
                selfplay_stats=selfplay_stats,
                logged_metrics=metrics,
            )
        text = buffer.getvalue()
        self.assertIn("WARN value loss domina", text)
        self.assertIn("WARN avg_turns=90", text)
        self.assertIn("WARN cache hit rate baja", text)

    def test_log_eval_marks_best(self) -> None:
        monitor = TrainingMonitor(total_iterations=10, log_every=5)
        stats = {"wins": 6, "losses": 4, "draws": 0, "score": 0.6, "heuristic_level": "hard", "sims": 400}

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            is_best = monitor.log_eval(iteration=4, eval_stats=stats)
        text = buffer.getvalue()
        self.assertTrue(is_best)
        self.assertIn("*** BEST ***", text)

    def test_log_eval_composite_tracks_best_score(self) -> None:
        monitor = TrainingMonitor(total_iterations=20, log_every=5)
        level_scores = {"hard": 0.52, "apex": 0.48, "sentinel": 0.50}

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            is_best = monitor.log_eval_composite(iteration=7, level_scores=level_scores)
        text = buffer.getvalue()
        self.assertTrue(is_best)
        self.assertIn("EVAL_COMPOSITE score=0.500", text)
        self.assertIn("*** BEST ***", text)


if __name__ == "__main__":
    unittest.main()
