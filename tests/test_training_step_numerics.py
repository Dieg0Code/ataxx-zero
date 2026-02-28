from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.actions import ACTION_SPACE
from model.system import AtaxxZero


class TestTrainingStepNumerics(unittest.TestCase):
    """Tests del paso de entrenamiento para evitar NaNs en loss/gradientes.

    Nota didáctica:
    En modelos, no basta con que el `forward` sea finito.
    También hay que validar:
    1) la pérdida final,
    2) la retropropagación (gradientes),
    3) que ambos sean finitos.
    """

    def setUp(self) -> None:
        torch.manual_seed(21)

    def test_loss_and_gradients_are_finite(self) -> None:
        system = AtaxxZero(
            learning_rate=1e-3,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            scheduler_type="none",
        )
        system.train()

        batch_size = 8
        boards = torch.randn(batch_size, 3, 7, 7)
        policy_targets = torch.rand(batch_size, ACTION_SPACE.num_actions)
        policy_targets = policy_targets / torch.sum(policy_targets, dim=1, keepdim=True)
        value_targets = torch.rand(batch_size) * 2.0 - 1.0

        loss = system.training_step((boards, policy_targets, value_targets), batch_idx=0)
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()
        grads = [
            parameter.grad
            for parameter in system.parameters()
            if parameter.grad is not None
        ]
        self.assertGreater(len(grads), 0)
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all().item())

    def test_predict_step_accepts_tensor_and_tuple_batch(self) -> None:
        system = AtaxxZero(
            learning_rate=1e-3,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            scheduler_type="none",
        )
        system.eval()

        boards = torch.randn(4, 3, 7, 7)
        policy_a, value_a = system.predict_step(boards, batch_idx=0)
        policy_b, value_b = system.predict_step((boards,), batch_idx=0)

        self.assertEqual(policy_a.shape, (4, ACTION_SPACE.num_actions))
        self.assertEqual(value_a.shape, (4, 1))
        self.assertEqual(policy_b.shape, (4, ACTION_SPACE.num_actions))
        self.assertEqual(value_b.shape, (4, 1))

    def test_forward_passes_action_mask_to_inner_model(self) -> None:
        system = AtaxxZero(
            learning_rate=1e-3,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            scheduler_type="none",
        )
        boards = torch.randn(2, 3, 7, 7)
        mask = torch.ones(2, ACTION_SPACE.num_actions)

        original_forward = system.model.forward
        with patch.object(system.model, "forward", wraps=original_forward) as forward_spy:
            _ = system(boards, action_mask=mask)
            _, kwargs = forward_spy.call_args
            self.assertIn("action_mask", kwargs)
            action_mask_obj = kwargs["action_mask"]
            self.assertIsInstance(action_mask_obj, torch.Tensor)
            self.assertTrue(torch.equal(action_mask_obj, mask))


if __name__ == "__main__":
    unittest.main()
