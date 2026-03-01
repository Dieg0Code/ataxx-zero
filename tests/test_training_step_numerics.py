from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as functional

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
        boards = torch.randn(batch_size, 4, 7, 7)
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

        boards = torch.randn(4, 4, 7, 7)
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
        boards = torch.randn(2, 4, 7, 7)
        mask = torch.ones(2, ACTION_SPACE.num_actions)

        original_forward = system.model.forward
        with patch.object(system.model, "forward", wraps=original_forward) as forward_spy:
            _ = system(boards, action_mask=mask)
            _, kwargs = forward_spy.call_args
            self.assertIn("action_mask", kwargs)
            action_mask_obj = kwargs["action_mask"]
            self.assertIsInstance(action_mask_obj, torch.Tensor)
            self.assertTrue(torch.equal(action_mask_obj, mask))

    def test_common_step_uses_policy_support_as_action_mask(self) -> None:
        system = AtaxxZero(
            learning_rate=1e-3,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            scheduler_type="none",
        )
        boards = torch.randn(2, 4, 7, 7)
        target_pis = torch.zeros(2, ACTION_SPACE.num_actions)
        target_pis[0, 7] = 1.0
        target_vs = torch.zeros(2)
        pi_logits = torch.randn(2, ACTION_SPACE.num_actions)
        v_pred = torch.zeros(2, 1)

        with patch.object(system.model, "forward", return_value=(pi_logits, v_pred)) as forward_spy:
            _ = system._common_step((boards, target_pis, target_vs))
            _, kwargs = forward_spy.call_args

        self.assertIn("action_mask", kwargs)
        action_mask = kwargs["action_mask"]
        self.assertIsInstance(action_mask, torch.Tensor)
        self.assertEqual(action_mask.shape, target_pis.shape)
        self.assertEqual(float(action_mask[0, 7].item()), 1.0)
        self.assertEqual(float(torch.sum(action_mask[0]).item()), 1.0)
        self.assertTrue(torch.all(action_mask[1] == 1.0).item())

    def test_common_step_applies_value_loss_coefficient(self) -> None:
        system = AtaxxZero(
            learning_rate=1e-3,
            value_loss_coeff=0.5,
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            scheduler_type="none",
        )
        boards = torch.randn(2, 4, 7, 7)
        target_pis = torch.zeros(2, ACTION_SPACE.num_actions)
        target_pis[0, 0] = 1.0
        target_pis[1, 1] = 1.0
        target_vs = torch.tensor([1.0, -1.0], dtype=torch.float32)
        pi_logits = torch.zeros(2, ACTION_SPACE.num_actions)
        v_pred = torch.tensor([[0.0], [0.0]], dtype=torch.float32)

        with patch.object(system.model, "forward", return_value=(pi_logits, v_pred)):
            metrics = system._common_step((boards, target_pis, target_vs))

        expected_v = functional.mse_loss(v_pred.view(-1), target_vs.view(-1))
        expected_pi = -torch.sum(target_pis * functional.log_softmax(pi_logits, dim=1)) / 2.0
        expected_total = expected_pi + 0.5 * expected_v

        self.assertTrue(torch.isclose(metrics["loss_value"], expected_v))
        self.assertTrue(torch.isclose(metrics["loss_policy"], expected_pi))
        self.assertTrue(torch.isclose(metrics["loss"], expected_total))


if __name__ == "__main__":
    unittest.main()
