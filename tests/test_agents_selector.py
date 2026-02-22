from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.selector import pick_ai_move
from engine.mcts import MCTS
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from model.transformer import AtaxxTransformerNet


class TestAgentsSelector(unittest.TestCase):
    """Regresion de agentes tras refactors de UI y estructura."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(123)

    def test_returns_pass_when_no_legal_moves(self) -> None:
        board = AtaxxBoard()
        board.grid[:, :] = 1
        board.current_player = 1

        move, text = pick_ai_move(
            board=board,
            agent="heuristic",
            rng=self.rng,
            heuristic_level="normal",
            mcts=None,
        )
        self.assertIsNone(move)
        self.assertIn("passed", text)

    def test_random_move_is_legal(self) -> None:
        board = AtaxxBoard()
        move, _ = pick_ai_move(
            board=board,
            agent="random",
            rng=self.rng,
            heuristic_level="normal",
            mcts=None,
        )
        self.assertIn(move, board.get_valid_moves())

    def test_heuristic_levels_return_legal_moves(self) -> None:
        board = AtaxxBoard()
        for level in ("easy", "normal", "hard"):
            move, _ = pick_ai_move(
                board=board,
                agent="heuristic",
                rng=self.rng,
                heuristic_level=level,
                mcts=None,
            )
            self.assertIn(move, board.get_valid_moves(), msg=f"Illegal move at level={level}")

    def test_model_agent_requires_mcts(self) -> None:
        board = AtaxxBoard()
        with self.assertRaises(RuntimeError):
            pick_ai_move(
                board=board,
                agent="model",
                rng=self.rng,
                heuristic_level="normal",
                mcts=None,
            )

    def test_model_agent_returns_legal_move(self) -> None:
        board = AtaxxBoard()
        model = AtaxxTransformerNet(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )
        mcts = MCTS(model=model, c_puct=1.5, n_simulations=8, device="cpu")
        move, _ = pick_ai_move(
            board=board,
            agent="model",
            rng=self.rng,
            heuristic_level="normal",
            mcts=mcts,
        )
        self.assertIn(move, board.get_valid_moves())
        self.assertIn(ACTION_SPACE.encode(move), range(ACTION_SPACE.num_actions))

    def test_unsupported_agent_raises(self) -> None:
        board = AtaxxBoard()
        with self.assertRaises(ValueError):
            pick_ai_move(
                board=board,
                agent="invalid",  # type: ignore[arg-type]
                rng=self.rng,
                heuristic_level="normal",
                mcts=None,
            )


if __name__ == "__main__":
    unittest.main()
