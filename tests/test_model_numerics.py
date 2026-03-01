from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import cast

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
        x = torch.randn(8, 4, 7, 7)
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
        x = torch.randn(4, 4, 7, 7)
        policy, value = model.predict(x)

        sums = torch.sum(policy, dim=1)
        self.assertTrue(torch.isfinite(policy).all().item())
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_policy_head_output_shape_spatial(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        model.eval()
        x = torch.zeros(2, 4, 7, 7)
        with torch.no_grad():
            logits, value = model(x)
        self.assertEqual(tuple(logits.shape), (2, ACTION_SPACE.num_actions))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_action_cell_indices_are_valid(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        src_idx = cast(torch.Tensor, model._action_src_idx)
        dst_idx = cast(torch.Tensor, model._action_dst_idx)
        self.assertEqual(tuple(src_idx.shape), (ACTION_SPACE.num_actions,))
        self.assertEqual(tuple(dst_idx.shape), (ACTION_SPACE.num_actions,))
        self.assertGreaterEqual(int(src_idx.min().item()), 0)
        self.assertLess(int(src_idx.max().item()), 49)
        self.assertGreaterEqual(int(dst_idx.min().item()), 0)
        self.assertLess(int(dst_idx.max().item()), 49)

    def test_policy_logits_masked_correctly(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        model.eval()
        x = torch.zeros(1, 4, 7, 7)
        mask = torch.zeros(1, ACTION_SPACE.num_actions)
        mask[0, 0] = 1.0
        with torch.no_grad():
            logits, _ = model(x, action_mask=mask)
        self.assertTrue(torch.isfinite(logits[0, 0]).item())
        min_val = torch.finfo(logits.dtype).min
        self.assertEqual(float(logits[0, 1].item()), float(min_val))

    def test_eval_determinism_spatial_head(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        model.eval()
        x = torch.randn(1, 4, 7, 7)
        with torch.no_grad():
            logits_1, value_1 = model(x)
            logits_2, value_2 = model(x)
        self.assertTrue(torch.allclose(logits_1, logits_2))
        self.assertTrue(torch.allclose(value_1, value_2))


if __name__ == "__main__":
    unittest.main()
