from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.actions import ACTION_SPACE
from model.transformer import AtaxxTransformerNet


class TestModelNumerics(unittest.TestCase):
    """Tests de estabilidad numérica del modelo.

    Nota didáctica:
    Estos tests son clave para detectar NaNs tempranamente, antes de gastar horas
    de GPU en entrenamiento.
    """

    def setUp(self) -> None:
        torch.manual_seed(7)
        np.random.seed(7)

    def test_forward_outputs_are_finite(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        x = torch.randn(8, 3, 7, 7)
        logits, value = model(x)

        self.assertEqual(tuple(logits.shape), (8, ACTION_SPACE.num_actions))
        self.assertEqual(tuple(value.shape), (8, 1))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(value).all().item())

    def test_predict_policy_is_normalized_and_finite(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        x = torch.randn(4, 3, 7, 7)
        policy, value = model.predict(x)

        sums = torch.sum(policy, dim=1)
        self.assertTrue(torch.isfinite(policy).all().item())
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
