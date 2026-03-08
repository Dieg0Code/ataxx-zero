from __future__ import annotations

import unittest

import numpy as np

from data.replay_buffer import sample_recent_mix
from game.actions import ACTION_SPACE


class TestReplayBufferNumerics(unittest.TestCase):
    def test_sample_recent_mix_biases_toward_recent_examples(self) -> None:
        examples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for idx in range(100):
            obs = np.zeros((4, 7, 7), dtype=np.float32)
            obs[0, 0, 0] = float(idx)
            pi = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
            pi[0] = 1.0
            examples.append((obs, pi, 0.0))

        sampled = sample_recent_mix(
            examples,
            recent_fraction=0.8,
            recent_window_fraction=0.2,
            seed=7,
            sample_size=100,
        )
        markers = [float(obs[0, 0, 0]) for obs, _, _ in sampled]
        recent_hits = sum(1 for marker in markers if marker >= 80.0)
        self.assertGreaterEqual(recent_hits, 55)

    def test_sample_recent_mix_is_deterministic_for_seed(self) -> None:
        examples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for idx in range(20):
            obs = np.zeros((4, 7, 7), dtype=np.float32)
            obs[0, 0, 0] = float(idx)
            pi = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
            pi[0] = 1.0
            examples.append((obs, pi, 0.0))

        sample_a = sample_recent_mix(
            examples,
            recent_fraction=0.7,
            recent_window_fraction=0.4,
            seed=123,
            sample_size=20,
        )
        sample_b = sample_recent_mix(
            examples,
            recent_fraction=0.7,
            recent_window_fraction=0.4,
            seed=123,
            sample_size=20,
        )
        markers_a = [float(obs[0, 0, 0]) for obs, _, _ in sample_a]
        markers_b = [float(obs[0, 0, 0]) for obs, _, _ in sample_b]
        self.assertListEqual(markers_a, markers_b)


if __name__ == "__main__":
    unittest.main()
