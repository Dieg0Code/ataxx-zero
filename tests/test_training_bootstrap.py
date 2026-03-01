from __future__ import annotations

import unittest

import numpy as np

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE
from training.bootstrap import generate_imitation_data, history_to_examples


class TestTrainingBootstrap(unittest.TestCase):
    def test_generate_imitation_data_produces_valid_examples(self) -> None:
        examples = generate_imitation_data(
            n_games=3,
            heuristic_level="hard",
            seed=123,
        )

        self.assertGreater(len(examples), 0)
        for observation, policy, value in examples:
            self.assertEqual(observation.shape, (4, BOARD_SIZE, BOARD_SIZE))
            self.assertEqual(policy.shape, (ACTION_SPACE.num_actions,))
            self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=6)
            self.assertEqual(int(np.count_nonzero(policy)), 1)
            self.assertIn(value, (-1.0, 0.0, 1.0))

    def test_history_to_examples_assigns_value_by_player_perspective(self) -> None:
        obs = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        policy[ACTION_SPACE.pass_index] = 1.0
        history = [
            (obs.copy(), policy.copy(), 1),
            (obs.copy(), policy.copy(), -1),
        ]

        p1_win = history_to_examples(history, winner=1)
        self.assertEqual(p1_win[0][2], 1.0)
        self.assertEqual(p1_win[1][2], -1.0)

        p2_win = history_to_examples(history, winner=-1)
        self.assertEqual(p2_win[0][2], -1.0)
        self.assertEqual(p2_win[1][2], 1.0)

        draw = history_to_examples(history, winner=0)
        self.assertEqual(draw[0][2], 0.0)
        self.assertEqual(draw[1][2], 0.0)


if __name__ == "__main__":
    unittest.main()
