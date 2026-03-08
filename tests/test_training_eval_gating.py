from __future__ import annotations

import unittest

from training.eval_gating import compute_regression_gate


class TestTrainingEvalGating(unittest.TestCase):
    def test_regression_streak_increments_when_score_drops_below_margin(self) -> None:
        streak, should_restore = compute_regression_gate(
            current_score=0.42,
            best_score=0.50,
            regression_delta=0.03,
            current_streak=1,
            patience=3,
        )
        self.assertEqual(streak, 2)
        self.assertFalse(should_restore)

    def test_regression_streak_resets_when_score_is_near_best(self) -> None:
        streak, should_restore = compute_regression_gate(
            current_score=0.48,
            best_score=0.50,
            regression_delta=0.03,
            current_streak=2,
            patience=3,
        )
        self.assertEqual(streak, 0)
        self.assertFalse(should_restore)

    def test_gate_triggers_restore_when_patience_is_reached(self) -> None:
        streak, should_restore = compute_regression_gate(
            current_score=0.40,
            best_score=0.50,
            regression_delta=0.02,
            current_streak=1,
            patience=2,
        )
        self.assertEqual(streak, 2)
        self.assertTrue(should_restore)


if __name__ == "__main__":
    unittest.main()
