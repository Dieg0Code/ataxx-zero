from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine.mcts import MCTS
from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS
from training.selfplay_episode_runtime import play_episode


class _FakeMCTS:
    def advance_root(self, root: object | None, action_idx: int) -> tuple[object | None, int]:
        return root, action_idx


class _ScriptedBoard:
    def __init__(self) -> None:
        self.current_player = 1
        self._turn = 0

    def is_game_over(self) -> bool:
        return self._turn >= 2

    def get_observation(self) -> np.ndarray:
        marker = float((self._turn * 10) + (1 if self.current_player == 1 else 2))
        return np.full((OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE), marker, dtype=np.float32)

    def step(self, move: tuple[int, int, int, int] | None) -> None:
        self._turn += 1
        self.current_player *= -1

    def get_result(self) -> int:
        return 1

    def is_forced_draw(self) -> bool:
        return False


class TestTrainingSelfplayEpisodeRuntime(unittest.TestCase):
    def test_play_episode_records_heuristic_turn_as_teacher_policy(self) -> None:
        model_move = (0, 0, 1, 1)
        heuristic_move = (6, 6, 5, 5)
        model_idx = ACTION_SPACE.encode(model_move)
        heuristic_idx = ACTION_SPACE.encode(heuristic_move)
        model_probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        model_probs[model_idx] = 1.0

        with patch("game.board.AtaxxBoard", _ScriptedBoard), patch(
            "training.selfplay_episode_runtime.compute_action_probs",
            return_value=(model_probs, None),
        ), patch(
            "training.selfplay_episode_runtime.heuristic_move",
            return_value=heuristic_move,
        ), patch(
            "training.selfplay_episode_runtime.cfg_bool",
            return_value=False,
        ):
            history, winner, turn_idx, forced_draw = play_episode(
                mcts=cast(MCTS, _FakeMCTS()),
                add_noise=False,
                temp_threshold=0,
                rng=np.random.default_rng(seed=7),
                opponent_type="heuristic",
                opponent_heuristic_level="sentinel",
                model_player=1,
            )

        self.assertEqual(winner, 1)
        self.assertEqual(turn_idx, 2)
        self.assertFalse(forced_draw)
        self.assertEqual(len(history), 2)

        model_observation, model_policy, model_player = history[0][:3]
        heuristic_observation, heuristic_policy, heuristic_player = history[1][:3]

        self.assertEqual(model_player, 1)
        self.assertEqual(heuristic_player, -1)
        self.assertEqual(float(model_observation[0, 0, 0]), 1.0)
        self.assertEqual(float(heuristic_observation[0, 0, 0]), 12.0)
        self.assertEqual(int(np.argmax(model_policy)), model_idx)
        self.assertEqual(int(np.argmax(heuristic_policy)), heuristic_idx)
        self.assertAlmostEqual(float(np.sum(heuristic_policy)), 1.0, places=6)

    def test_play_episode_records_checkpoint_turn_as_teacher_policy(self) -> None:
        model_move = (0, 0, 1, 1)
        checkpoint_move = (6, 6, 5, 5)
        model_idx = ACTION_SPACE.encode(model_move)
        checkpoint_idx = ACTION_SPACE.encode(checkpoint_move)
        model_probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        checkpoint_probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        model_probs[model_idx] = 1.0
        checkpoint_probs[checkpoint_idx] = 1.0

        with patch("game.board.AtaxxBoard", _ScriptedBoard), patch(
            "training.selfplay_episode_runtime.compute_action_probs",
            side_effect=[(model_probs, None), (checkpoint_probs, None)],
        ), patch(
            "training.selfplay_episode_runtime.cfg_bool",
            return_value=False,
        ):
            history, winner, turn_idx, forced_draw = play_episode(
                mcts=cast(MCTS, _FakeMCTS()),
                add_noise=False,
                temp_threshold=0,
                rng=np.random.default_rng(seed=7),
                opponent_type="checkpoint",
                opponent_heuristic_level="sentinel",
                model_player=1,
                opponent_checkpoint_mcts=cast(MCTS, _FakeMCTS()),
            )

        self.assertEqual(winner, 1)
        self.assertEqual(turn_idx, 2)
        self.assertFalse(forced_draw)
        self.assertEqual(len(history), 2)

        model_observation, model_policy, model_player = history[0][:3]
        checkpoint_observation, checkpoint_policy, checkpoint_player = history[1][:3]

        self.assertEqual(model_player, 1)
        self.assertEqual(checkpoint_player, -1)
        self.assertEqual(float(model_observation[0, 0, 0]), 1.0)
        self.assertEqual(float(checkpoint_observation[0, 0, 0]), 12.0)
        self.assertEqual(int(np.argmax(model_policy)), model_idx)
        self.assertEqual(int(np.argmax(checkpoint_policy)), checkpoint_idx)
        self.assertAlmostEqual(float(np.sum(checkpoint_policy)), 1.0, places=6)

    def test_checkpoint_opponent_uses_opening_temperature_schedule(self) -> None:
        model_move = (0, 0, 1, 1)
        checkpoint_move = (6, 6, 5, 5)
        model_idx = ACTION_SPACE.encode(model_move)
        checkpoint_idx = ACTION_SPACE.encode(checkpoint_move)
        model_probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        checkpoint_probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        model_probs[model_idx] = 1.0
        checkpoint_probs[checkpoint_idx] = 1.0

        with patch("game.board.AtaxxBoard", _ScriptedBoard), patch(
            "training.selfplay_episode_runtime.compute_action_probs",
            side_effect=[(model_probs, None), (checkpoint_probs, None)],
        ) as compute_mock, patch(
            "training.selfplay_episode_runtime.select_action_idx",
            side_effect=[model_idx, checkpoint_idx],
        ), patch(
            "training.selfplay_episode_runtime.cfg_bool",
            return_value=False,
        ):
            play_episode(
                mcts=cast(MCTS, _FakeMCTS()),
                add_noise=False,
                temp_threshold=4,
                rng=np.random.default_rng(seed=7),
                opponent_type="checkpoint",
                opponent_heuristic_level="sentinel",
                model_player=1,
                opponent_checkpoint_mcts=cast(MCTS, _FakeMCTS()),
            )

        temperatures = [float(call.kwargs["temperature"]) for call in compute_mock.call_args_list]
        self.assertEqual(temperatures, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
