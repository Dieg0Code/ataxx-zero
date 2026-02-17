from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
