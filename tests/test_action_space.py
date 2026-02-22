from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.actions import ACTION_SPACE


class TestActionSpace(unittest.TestCase):
    """Tests para el mapeo acción <-> movimiento.

    Nota didáctica:
    En AlphaZero-like, si este mapeo está mal, todo el entrenamiento de política
    queda roto (target pi y logits apuntan a acciones diferentes).
    """

    def test_encode_decode_roundtrip_for_regular_moves(self) -> None:
        # Probamos un subconjunto para no hacer test pesado.
        checked = 0
        for idx, move in enumerate(ACTION_SPACE.idx_to_move):
            if move is None:
                continue
            encoded = ACTION_SPACE.encode(move)
            decoded = ACTION_SPACE.decode(encoded)
            self.assertEqual(encoded, idx)
            self.assertEqual(decoded, move)
            checked += 1
            if checked >= 200:
                break
        self.assertGreater(checked, 0)

    def test_pass_index_is_consistent(self) -> None:
        pass_idx = ACTION_SPACE.encode(None)
        decoded = ACTION_SPACE.decode(pass_idx)
        self.assertEqual(pass_idx, ACTION_SPACE.pass_index)
        self.assertIsNone(decoded)

    def test_mask_from_moves_can_include_pass(self) -> None:
        mask = ACTION_SPACE.mask_from_moves(moves=[], include_pass=True)
        self.assertEqual(mask.shape[0], ACTION_SPACE.num_actions)
        self.assertEqual(float(mask[ACTION_SPACE.pass_index]), 1.0)
        self.assertAlmostEqual(float(mask.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
