from __future__ import annotations

import unittest

from inference.checkpoint_duel_runtime import (
    build_match_schedule,
    summarize_match_results,
)


class TestInferenceCheckpointDuelRuntime(unittest.TestCase):
    def test_build_match_schedule_alternates_colors(self) -> None:
        schedule = build_match_schedule(games=6)

        self.assertEqual(len(schedule), 6)
        self.assertEqual(schedule[0], (1, -1))
        self.assertEqual(schedule[1], (-1, 1))
        self.assertEqual(schedule[2], (1, -1))
        self.assertEqual(schedule[3], (-1, 1))

    def test_summarize_match_results_counts_checkpoint_a_perspective(self) -> None:
        summary = summarize_match_results(
            results=[
                {"winner": 1, "checkpoint_a_player": 1, "turns": 30},
                {"winner": -1, "checkpoint_a_player": 1, "turns": 42},
                {"winner": 0, "checkpoint_a_player": -1, "turns": 55},
                {"winner": -1, "checkpoint_a_player": -1, "turns": 28},
            ],
        )

        self.assertEqual(summary["games"], 4)
        self.assertEqual(summary["checkpoint_a_wins"], 2)
        self.assertEqual(summary["checkpoint_b_wins"], 1)
        self.assertEqual(summary["draws"], 1)
        self.assertAlmostEqual(float(summary["checkpoint_a_score"]), 0.625)
        self.assertAlmostEqual(float(summary["avg_turns"]), 38.75)


if __name__ == "__main__":
    unittest.main()
