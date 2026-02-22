from __future__ import annotations

import numpy as np

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.types import Move


def model_move(board: AtaxxBoard, mcts: object | None) -> Move | None:
    from engine.mcts import MCTS

    if not isinstance(mcts, MCTS):
        raise RuntimeError("Model agent selected but MCTS is not initialized.")
    probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
    action_idx = int(np.argmax(probs))
    return ACTION_SPACE.decode(action_idx)
