from __future__ import annotations

import unittest

import numpy as np

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS
from training.config_runtime import CONFIG
from training.reward_runtime import (
    compute_state_potential,
    compute_transition_shaping_reward,
    history_to_examples,
)


class TestTrainingRewardRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_transition_shaping_reward_is_positive_for_stronger_position(self) -> None:
        board = AtaxxBoard()
        acting_player = int(board.current_player)
        before_potential = compute_state_potential(board, acting_player)

        board.step((0, 0, 1, 1))
        after_potential = compute_state_potential(board, acting_player)
        reward = compute_transition_shaping_reward(
            before_potential=before_potential,
            after_potential=after_potential,
        )

        self.assertGreater(reward, 0.0)

    def test_history_to_examples_keeps_terminal_targets_when_shaping_is_disabled(self) -> None:
        CONFIG["reward_shaping_enabled"] = False
        obs = np.zeros((OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        policy[ACTION_SPACE.pass_index] = 1.0
        history = [
            (obs.copy(), policy.copy(), 1, 0.4),
            (obs.copy(), policy.copy(), -1, 0.1),
        ]

        examples = history_to_examples(history, winner=1)

        self.assertEqual(examples[0][2], 1.0)
        self.assertEqual(examples[1][2], -1.0)

    def test_history_to_examples_accumulates_opponent_rewards_with_sign_flip(self) -> None:
        CONFIG["reward_shaping_enabled"] = True
        CONFIG["reward_shaping_scale"] = 0.5
        CONFIG["reward_shaping_gamma"] = 1.0
        CONFIG["reward_shaping_draw_penalty"] = 0.0
        obs = np.zeros((OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        policy[ACTION_SPACE.pass_index] = 1.0
        history = [
            (obs.copy(), policy.copy(), 1, 0.4),
            (obs.copy(), policy.copy(), -1, 0.1),
        ]

        examples = history_to_examples(history, winner=0)

        self.assertAlmostEqual(examples[0][2], 0.15, places=6)
        self.assertAlmostEqual(examples[1][2], 0.05, places=6)

    def test_history_to_examples_accumulates_same_player_rewards_positively(self) -> None:
        CONFIG["reward_shaping_enabled"] = True
        CONFIG["reward_shaping_scale"] = 0.5
        CONFIG["reward_shaping_gamma"] = 1.0
        CONFIG["reward_shaping_draw_penalty"] = 0.0
        obs = np.zeros((OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        policy[ACTION_SPACE.pass_index] = 1.0
        history = [
            (obs.copy(), policy.copy(), 1, 0.4),
            (obs.copy(), policy.copy(), 1, 0.2),
        ]

        examples = history_to_examples(history, winner=0)

        self.assertAlmostEqual(examples[0][2], 0.3, places=6)
        self.assertAlmostEqual(examples[1][2], 0.1, places=6)

    def test_history_to_examples_penalizes_forced_draws(self) -> None:
        CONFIG["reward_shaping_enabled"] = False
        CONFIG["reward_shaping_draw_penalty"] = 0.2
        obs = np.zeros((OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        policy[ACTION_SPACE.pass_index] = 1.0
        history = [
            (obs.copy(), policy.copy(), 1),
            (obs.copy(), policy.copy(), -1),
        ]

        examples = history_to_examples(history, winner=0, forced_draw=True)

        self.assertAlmostEqual(examples[0][2], -0.2, places=6)
        self.assertAlmostEqual(examples[1][2], -0.2, places=6)


if __name__ == "__main__":
    unittest.main()
