from __future__ import annotations

import unittest

import numpy as np

from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2
from game.serialization import board_from_state, board_to_state


class TestBoardSerialization(unittest.TestCase):
    def test_roundtrip_preserves_state(self) -> None:
        board = AtaxxBoard()
        board.step((0, 0, 1, 1))
        board.half_moves = 17

        payload = board_to_state(board)
        loaded = board_from_state(payload)

        self.assertEqual(int(loaded.current_player), int(board.current_player))
        self.assertEqual(int(loaded.half_moves), 17)
        self.assertTrue(np.array_equal(loaded.grid, board.grid))

    def test_rejects_invalid_grid_shape(self) -> None:
        payload = {
            "grid": [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE - 1)],
            "current_player": PLAYER_1,
            "half_moves": 0,
        }
        with self.assertRaises(ValueError):
            board_from_state(payload)

    def test_rejects_invalid_cell_value(self) -> None:
        grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[0][0] = 9
        payload = {
            "grid": grid,
            "current_player": PLAYER_1,
            "half_moves": 0,
        }
        with self.assertRaises(ValueError):
            board_from_state(payload)

    def test_rejects_invalid_player(self) -> None:
        payload = {
            "grid": [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)],
            "current_player": 0,
            "half_moves": 0,
        }
        with self.assertRaises(ValueError):
            board_from_state(payload)

    def test_rejects_negative_half_moves(self) -> None:
        payload = {
            "grid": [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)],
            "current_player": PLAYER_2,
            "half_moves": -1,
        }
        with self.assertRaises(ValueError):
            board_from_state(payload)

    def test_rejects_missing_required_keys(self) -> None:
        payload = {
            "grid": [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)],
            "current_player": PLAYER_1,
        }
        with self.assertRaises(ValueError):
            board_from_state(payload)


if __name__ == "__main__":
    unittest.main()
