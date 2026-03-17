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
        board._position_counts[board._position_key()] = 2

        payload = board_to_state(board)
        loaded = board_from_state(payload)

        self.assertEqual(int(loaded.current_player), int(board.current_player))
        self.assertEqual(int(loaded.half_moves), 17)
        self.assertTrue(np.array_equal(loaded.grid, board.grid))
        self.assertEqual(loaded._position_counts, board._position_counts)

    def test_roundtrip_preserves_non_current_repetition_history(self) -> None:
        board = AtaxxBoard()
        previous_board = board.copy()
        board.step((0, 0, 1, 1))
        board.step((0, 6, 1, 5))
        board._position_counts[previous_board._position_key()] = 2

        loaded = board_from_state(board_to_state(board))

        self.assertEqual(
            loaded._position_counts[previous_board._position_key()],
            2,
        )

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

    def test_rejects_position_counts_without_current_board_state(self) -> None:
        board = AtaxxBoard()
        payload = board_to_state(board)
        payload["position_counts"] = [
            {
                "grid": [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)],
                "current_player": PLAYER_1,
                "count": 1,
            }
        ]
        with self.assertRaises(ValueError):
            board_from_state(payload)

    def test_board_to_state_normalizes_stale_repetition_history(self) -> None:
        board = AtaxxBoard()
        board.grid[:, :] = PLAYER_1
        board.current_player = PLAYER_1

        payload = board_to_state(board)

        self.assertEqual(len(payload["position_counts"]), 1)
        self.assertEqual(payload["position_counts"][0]["current_player"], PLAYER_1)
        self.assertEqual(payload["position_counts"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
