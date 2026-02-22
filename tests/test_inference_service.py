from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from inference.service import InferenceService
from model.system import AtaxxZero


class TestInferenceService(unittest.TestCase):
    """Tests basicos del servicio de inferencia para API."""

    def _tiny_system(self) -> AtaxxZero:
        return AtaxxZero(
            d_model=64,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
        )

    def test_fast_mode_returns_legal_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            system = self._tiny_system()
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"state_dict": system.state_dict()}, ckpt_path)

            service = InferenceService(
                checkpoint_path=ckpt_path,
                device="cpu",
                model_kwargs={
                    "d_model": 64,
                    "nhead": 8,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.0,
                },
            )
            board = AtaxxBoard()
            result = service.predict(board, mode="fast")

            legal_moves = board.get_valid_moves()
            legal_idxs = {ACTION_SPACE.encode(mv) for mv in legal_moves}
            self.assertEqual(result.mode, "fast")
            self.assertIn(result.action_idx, legal_idxs)
            self.assertIn(result.move, legal_moves)
            self.assertTrue(-1.0 <= result.value <= 1.0)

    def test_strong_mode_returns_legal_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            system = self._tiny_system()
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"state_dict": system.state_dict()}, ckpt_path)

            service = InferenceService(
                checkpoint_path=ckpt_path,
                device="cpu",
                mcts_sims=8,
                model_kwargs={
                    "d_model": 64,
                    "nhead": 8,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.0,
                },
            )
            board = AtaxxBoard()
            result = service.predict(board, mode="strong")

            legal_moves = board.get_valid_moves()
            legal_idxs = {ACTION_SPACE.encode(mv) for mv in legal_moves}
            self.assertEqual(result.mode, "strong")
            self.assertIn(result.action_idx, legal_idxs)
            self.assertIn(result.move, legal_moves)

    def test_game_over_returns_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            system = self._tiny_system()
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"state_dict": system.state_dict()}, ckpt_path)

            service = InferenceService(
                checkpoint_path=ckpt_path,
                device="cpu",
                model_kwargs={
                    "d_model": 64,
                    "nhead": 8,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.0,
                },
            )
            board = AtaxxBoard()
            board.grid[:, :] = 1
            board.current_player = 1
            board.half_moves = 1

            result = service.predict(board, mode="fast")
            self.assertIsNone(result.move)
            self.assertEqual(result.action_idx, ACTION_SPACE.pass_index)

    def test_rejects_invalid_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "invalid.pt"
            torch.save({"weights": {}}, ckpt_path)
            with self.assertRaises(ValueError):
                InferenceService(checkpoint_path=ckpt_path, device="cpu")

    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(FileNotFoundError):
            InferenceService(checkpoint_path="does/not/exist/model.pt", device="cpu")

    def test_rejects_invalid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            system = self._tiny_system()
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"state_dict": system.state_dict()}, ckpt_path)
            service = InferenceService(
                checkpoint_path=ckpt_path,
                device="cpu",
                model_kwargs={
                    "d_model": 64,
                    "nhead": 8,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.0,
                },
            )
            with self.assertRaises(ValueError):
                service.predict(AtaxxBoard(), mode="invalid")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
