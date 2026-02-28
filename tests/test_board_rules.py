from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2


class TestBoardRules(unittest.TestCase):
    """Tests de reglas base del juego.

    Nota didáctica:
    Estos tests son de "reglas duras" (invariantes).
    Si fallan, el entrenamiento se contamina porque los datos de self-play
    representan un juego incorrecto.
    """

    def test_initial_setup_has_expected_corner_pieces(self) -> None:
        board = AtaxxBoard()
        self.assertEqual(int(board.grid[0, 0]), PLAYER_1)
        self.assertEqual(int(board.grid[BOARD_SIZE - 1, BOARD_SIZE - 1]), PLAYER_1)
        self.assertEqual(int(board.grid[0, BOARD_SIZE - 1]), PLAYER_2)
        self.assertEqual(int(board.grid[BOARD_SIZE - 1, 0]), PLAYER_2)

    def test_pass_is_illegal_when_moves_exist(self) -> None:
        board = AtaxxBoard()
        with self.assertRaises(ValueError):
            board.step(None)

    def test_infection_converts_adjacent_enemy_pieces(self) -> None:
        # Escenario controlado para aislar la regla de infección.
        grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        grid[3, 3] = PLAYER_1
        grid[4, 4] = PLAYER_2
        board = AtaxxBoard(grid=grid, player=PLAYER_1)

        # Movimiento de clon adyacente al enemigo -> debe convertir (4,4).
        board.step((3, 3, 3, 4))
        self.assertEqual(int(board.grid[4, 4]), PLAYER_1)

    def test_observation_has_three_channels_without_nans(self) -> None:
        board = AtaxxBoard()
        obs = board.get_observation()
        self.assertEqual(obs.shape, (3, BOARD_SIZE, BOARD_SIZE))
        self.assertFalse(np.isnan(obs).any())
        self.assertTrue((obs >= 0).all())
        self.assertTrue((obs <= 1).all())
        self.assertTrue((board.grid == EMPTY).sum() > 0)

    def test_copy_creates_independent_board(self) -> None:
        board = AtaxxBoard()
        copied = board.copy()
        copied.step((0, 0, 1, 1))

        self.assertNotEqual(int(board.grid[1, 1]), int(copied.grid[1, 1]))
        self.assertNotEqual(board.half_moves, copied.half_moves)

    def test_valid_moves_matches_bruteforce_scan(self) -> None:
        rng = np.random.default_rng(7)
        grid = np.asarray(
            rng.choice(
                [EMPTY, PLAYER_1, PLAYER_2],
                size=(BOARD_SIZE, BOARD_SIZE),
                p=[0.5, 0.25, 0.25],
            ),
            dtype=np.int8,
        )
        board = AtaxxBoard(grid=grid, player=PLAYER_1)

        brute_moves: list[tuple[int, int, int, int]] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board.grid[r, c] != PLAYER_1:
                    continue
                r_min = max(0, r - 2)
                r_max = min(BOARD_SIZE, r + 3)
                c_min = max(0, c - 2)
                c_max = min(BOARD_SIZE, c + 3)
                for tr in range(r_min, r_max):
                    for tc in range(c_min, c_max):
                        if (tr != r or tc != c) and board.grid[tr, tc] == EMPTY:
                            brute_moves.append((r, c, tr, tc))

        self.assertEqual(set(board.get_valid_moves(player=PLAYER_1)), set(brute_moves))

    def test_copy_from_overwrites_state_in_place(self) -> None:
        source = AtaxxBoard()
        source.step((0, 0, 1, 1))
        target = AtaxxBoard()

        target.copy_from(source)

        self.assertTrue(np.array_equal(target.grid, source.grid))
        self.assertEqual(target.current_player, source.current_player)
        self.assertEqual(target.half_moves, source.half_moves)

    def test_piece_counters_stay_consistent_after_moves(self) -> None:
        board = AtaxxBoard()
        sequence = [
            (0, 0, 1, 1),
            (0, BOARD_SIZE - 1, 1, BOARD_SIZE - 2),
            (BOARD_SIZE - 1, BOARD_SIZE - 1, BOARD_SIZE - 2, BOARD_SIZE - 2),
            (BOARD_SIZE - 1, 0, BOARD_SIZE - 2, 1),
        ]
        for move in sequence:
            board.step(move)
            self.assertEqual(board.p1_count, int(np.sum(board.grid == PLAYER_1)))
            self.assertEqual(board.p2_count, int(np.sum(board.grid == PLAYER_2)))
            self.assertEqual(board.empty_count, int(np.sum(board.grid == EMPTY)))


if __name__ == "__main__":
    unittest.main()
