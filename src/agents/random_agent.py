from __future__ import annotations

import numpy as np

from game.board import AtaxxBoard
from game.types import Move


def random_move(board: AtaxxBoard, rng: np.random.Generator) -> Move | None:
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return None
    return moves[int(rng.integers(0, len(moves)))]
