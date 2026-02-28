from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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

    def test_mcts_amp_flag_is_disabled_on_cpu(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        mcts = MCTS(
            model=model,
            c_puct=1.5,
            n_simulations=8,
            device="cpu",
            use_amp=True,
        )
        board = AtaxxBoard()
        probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=1.0)

        self.assertFalse(mcts.use_amp)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_mcts_can_reuse_root_between_turns(self) -> None:
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        mcts = MCTS(model=model, c_puct=1.5, n_simulations=16, device="cpu")
        board = AtaxxBoard()

        probs, root = mcts.run_with_root(
            board=board,
            root=None,
            add_dirichlet_noise=False,
            temperature=0.0,
        )
        action_idx = int(np.argmax(probs))
        board.step(ACTION_SPACE.decode(action_idx))
        next_root = mcts.advance_root(root, action_idx)

        next_probs, _ = mcts.run_with_root(
            board=board,
            root=next_root,
            add_dirichlet_noise=False,
            temperature=1.0,
        )

        self.assertEqual(next_probs.shape, (ACTION_SPACE.num_actions,))
        self.assertFalse(np.isnan(next_probs).any())
        self.assertAlmostEqual(float(np.sum(next_probs)), 1.0, places=5)

    def test_mcts_inference_cache_reuses_model_forward(self) -> None:
        class CountingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(
                self,
                board_tensor: torch.Tensor,
                action_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.calls += 1
                batch = board_tensor.shape[0]
                logits = torch.zeros((batch, ACTION_SPACE.num_actions), dtype=torch.float32)
                value = torch.zeros((batch, 1), dtype=torch.float32)
                if action_mask is not None:
                    logits = logits.masked_fill(action_mask <= 0, -1e9)
                return logits, value

        model = CountingModel()
        mcts = MCTS(
            model=model,
            c_puct=1.5,
            n_simulations=0,
            device="cpu",
            cache_size=32,
        )
        board = AtaxxBoard()

        mcts.run(board=board, add_dirichlet_noise=False, temperature=1.0)
        first_calls = model.calls
        mcts.run(board=board, add_dirichlet_noise=False, temperature=1.0)
        second_calls = model.calls

        self.assertEqual(first_calls, 1)
        self.assertEqual(second_calls, 1)

    def test_mcts_leaf_batching_reduces_forward_calls(self) -> None:
        class CountingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(
                self,
                board_tensor: torch.Tensor,
                action_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.calls += 1
                batch = board_tensor.shape[0]
                logits = torch.zeros((batch, ACTION_SPACE.num_actions), dtype=torch.float32)
                value = torch.zeros((batch, 1), dtype=torch.float32)
                if action_mask is not None:
                    logits = logits.masked_fill(action_mask <= 0, -1e9)
                return logits, value

        board = AtaxxBoard()
        sims = 16

        model_no_batch = CountingModel()
        mcts_no_batch = MCTS(
            model=model_no_batch,
            c_puct=1.5,
            n_simulations=sims,
            device="cpu",
            cache_size=0,
            leaf_batch_size=1,
        )
        mcts_no_batch.run(board=board, add_dirichlet_noise=False, temperature=1.0)
        calls_no_batch = model_no_batch.calls

        model_batched = CountingModel()
        mcts_batched = MCTS(
            model=model_batched,
            c_puct=1.5,
            n_simulations=sims,
            device="cpu",
            cache_size=0,
            leaf_batch_size=8,
        )
        mcts_batched.run(board=board, add_dirichlet_noise=False, temperature=1.0)
        calls_batched = model_batched.calls

        self.assertLess(calls_batched, calls_no_batch)

    def test_mcts_cache_stats_report_hit_rate(self) -> None:
        class CountingModel(nn.Module):
            def forward(
                self,
                board_tensor: torch.Tensor,
                action_mask: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                batch = board_tensor.shape[0]
                logits = torch.zeros((batch, ACTION_SPACE.num_actions), dtype=torch.float32)
                value = torch.zeros((batch, 1), dtype=torch.float32)
                if action_mask is not None:
                    logits = logits.masked_fill(action_mask <= 0, -1e9)
                return logits, value

        model = CountingModel()
        mcts = MCTS(
            model=model,
            c_puct=1.5,
            n_simulations=4,
            device="cpu",
            cache_size=64,
            leaf_batch_size=4,
        )
        board = AtaxxBoard()
        mcts.run(board=board, add_dirichlet_noise=False, temperature=1.0)
        stats_first = mcts.cache_stats()
        mcts.run(board=board, add_dirichlet_noise=False, temperature=1.0)
        stats_second = mcts.cache_stats()

        self.assertGreaterEqual(int(stats_first["misses"]), 1)
        self.assertGreaterEqual(int(stats_second["hits"]), 1)
        self.assertGreaterEqual(float(stats_second["hit_rate"]), 0.0)
        self.assertLessEqual(float(stats_second["hit_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()
