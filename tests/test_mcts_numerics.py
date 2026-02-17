from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine.mcts import MCTS
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from model.transformer import AtaxxTransformerNet


class TestMCTSNumerics(unittest.TestCase):
    """Tests numéricos básicos de MCTS con modelo pequeño.

    Nota didáctica:
    MCTS combina red + softmax + normalizaciones. Es un punto común donde aparecen
    NaNs si hay máscaras inválidas o divisiones por cero.
    """

    def setUp(self) -> None:
        torch.manual_seed(11)
        np.random.seed(11)

    def test_mcts_returns_valid_distribution(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        mcts = MCTS(model=model, c_puct=1.5, n_simulations=32, device="cpu")
        board = AtaxxBoard()
        probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=1.0)

        self.assertEqual(probs.shape, (ACTION_SPACE.num_actions,))
        self.assertFalse(np.isnan(probs).any())
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue((probs >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
