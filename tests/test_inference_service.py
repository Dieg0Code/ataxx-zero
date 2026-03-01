from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
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

    def test_fast_mode_uses_onnx_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            system = self._tiny_system()
            ckpt_path = Path(tmp_dir) / "model.pt"
            onnx_path = Path(tmp_dir) / "model.onnx"
            onnx_path.write_bytes(b"fake")
            torch.save({"state_dict": system.state_dict()}, ckpt_path)

            fake_session = Mock()
            fake_session.get_inputs.return_value = [
                SimpleNamespace(name="board"),
                SimpleNamespace(name="action_mask"),
            ]
            fake_session.get_outputs.return_value = [
                SimpleNamespace(name="policy"),
                SimpleNamespace(name="value"),
            ]
            fake_logits = np.full((1, ACTION_SPACE.num_actions), -8.0, dtype=np.float32)
            fake_logits[0, ACTION_SPACE.encode((0, 0, 1, 1))] = 3.0
            fake_session.run.return_value = [fake_logits, np.array([[0.25]], dtype=np.float32)]

            with patch.object(InferenceService, "_load_onnx_session", return_value=fake_session):
                service = InferenceService(
                    checkpoint_path=ckpt_path,
                    onnx_path=onnx_path,
                    prefer_onnx=True,
                    device="cpu",
                    model_kwargs={
                        "d_model": 64,
                        "nhead": 8,
                        "num_layers": 2,
                        "dim_feedforward": 128,
                        "dropout": 0.0,
                    },
                )
                result = service.predict(AtaxxBoard(), mode="fast")

            self.assertEqual(result.action_idx, ACTION_SPACE.encode((0, 0, 1, 1)))
            self.assertEqual(result.mode, "fast")

    def test_strong_mode_degrades_to_fast_when_only_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            onnx_path.write_bytes(b"fake")

            fake_session = Mock()
            fake_session.get_inputs.return_value = [SimpleNamespace(name="board")]
            fake_session.get_outputs.return_value = [
                SimpleNamespace(name="policy"),
                SimpleNamespace(name="value"),
            ]
            fake_logits = np.full((1, ACTION_SPACE.num_actions), -7.0, dtype=np.float32)
            fake_logits[0, ACTION_SPACE.encode((0, 0, 1, 1))] = 2.0
            fake_session.run.return_value = [fake_logits, np.array([[0.1]], dtype=np.float32)]

            with patch.object(InferenceService, "_load_onnx_session", return_value=fake_session):
                service = InferenceService(
                    checkpoint_path=Path(tmp_dir) / "missing.ckpt",
                    onnx_path=onnx_path,
                    prefer_onnx=True,
                    device="cpu",
                )
                result = service.predict(AtaxxBoard(), mode="strong")

            self.assertEqual(result.mode, "fast")
            self.assertEqual(result.action_idx, ACTION_SPACE.encode((0, 0, 1, 1)))


if __name__ == "__main__":
    unittest.main()
