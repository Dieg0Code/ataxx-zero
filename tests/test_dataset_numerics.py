from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import AtaxxDataset
from data.replay_buffer import ReplayBuffer
from game.actions import ACTION_SPACE


class TestDatasetNumerics(unittest.TestCase):
    """Tests de dataset/augmentation orientados a estabilidad.

    Nota didáctica:
    Si augmentation rompe la normalización de `pi`, la loss de política puede
    explotar y generar NaNs durante entrenamiento.
    """

    def setUp(self) -> None:
        np.random.seed(33)
        self.buffer = ReplayBuffer(capacity=32)
        for _ in range(16):
            obs = np.random.randn(3, 7, 7).astype(np.float32)
            pi = np.random.rand(ACTION_SPACE.num_actions).astype(np.float32)
            pi /= float(np.sum(pi))
            value = float(np.random.choice([-1.0, 0.0, 1.0]))
            self.buffer.save_game([(obs, pi, value)])

    def test_augmented_sample_is_finite_and_normalized(self) -> None:
        dataset = AtaxxDataset(buffer=self.buffer, augment=True, reference_buffer=False)
        board, pi, value = dataset[0]

        self.assertIsInstance(board, torch.Tensor)
        self.assertIsInstance(pi, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertTrue(torch.isfinite(board).all().item())
        self.assertTrue(torch.isfinite(pi).all().item())
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertAlmostEqual(float(torch.sum(pi).item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
