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


if __name__ == "__main__":
    unittest.main()
