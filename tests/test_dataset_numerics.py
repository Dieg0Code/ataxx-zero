from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import AtaxxDataset, ValidationDataset
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
            obs = np.random.randn(4, 7, 7).astype(np.float32)
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

    def test_train_and_validation_splits_are_disjoint(self) -> None:
        split = 0.25
        buffer = ReplayBuffer(capacity=32)
        for idx in range(8):
            obs = np.zeros((4, 7, 7), dtype=np.float32)
            obs[0, 0, 0] = float(idx)
            pi = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
            pi[0] = 1.0
            buffer.save_game([(obs, pi, 0.0)])

        train_dataset = AtaxxDataset(
            buffer=buffer,
            augment=False,
            reference_buffer=False,
            val_split=split,
        )
        val_dataset = ValidationDataset(buffer=buffer, split=split)

        train_markers = {
            float(train_dataset[i][0][0, 0, 0].item())
            for i in range(len(train_dataset))
        }
        val_markers = {
            float(val_dataset[i][0][0, 0, 0].item())
            for i in range(len(val_dataset))
        }

        self.assertEqual(len(train_dataset), 6)
        self.assertEqual(len(val_dataset), 2)
        self.assertSetEqual(train_markers, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0})
        self.assertSetEqual(val_markers, {6.0, 7.0})
        self.assertTrue(train_markers.isdisjoint(val_markers))


if __name__ == "__main__":
    unittest.main()
