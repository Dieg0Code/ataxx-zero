from __future__ import annotations

import numpy as np

from .constants import BOARD_SIZE

Move = tuple[int, int, int, int]


class AtaxxActionSpace:
    """Fixed action indexing for policy head outputs."""

    def __init__(self, board_size: int = BOARD_SIZE) -> None:
        self.board_size = board_size
        self.idx_to_move: tuple[Move | None, ...]
        self.move_to_idx: dict[Move, int]
        self.pass_index: int
        self.num_actions: int

        deltas = [
            (dr, dc)
            for dr in range(-2, 3)
            for dc in range(-2, 3)
            if not (dr == 0 and dc == 0) and max(abs(dr), abs(dc)) <= 2
        ]

        idx_to_move_list: list[Move | None] = []
        move_to_idx: dict[Move, int] = {}

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr, dc in deltas:
                    tr = r + dr
                    tc = c + dc
                    if 0 <= tr < self.board_size and 0 <= tc < self.board_size:
                        move = (r, c, tr, tc)
                        move_to_idx[move] = len(idx_to_move_list)
                        idx_to_move_list.append(move)

        self.pass_index = len(idx_to_move_list)
        idx_to_move_list.append(None)

        self.idx_to_move = tuple(idx_to_move_list)
        self.move_to_idx = move_to_idx
        self.num_actions = len(self.idx_to_move)

    def encode(self, move: Move | None) -> int:
        if move is None:
            return self.pass_index
        return self.move_to_idx[move]

    def decode(self, action_idx: int) -> Move | None:
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"Invalid action index: {action_idx}")
        return self.idx_to_move[action_idx]

    def empty_mask(self) -> np.ndarray:
        return np.zeros(self.num_actions, dtype=np.float32)

    def mask_from_moves(self, moves: list[Move], include_pass: bool) -> np.ndarray:
        mask = self.empty_mask()
        for move in moves:
            mask[self.encode(move)] = 1.0
        if include_pass:
            mask[self.pass_index] = 1.0
        return mask


ACTION_SPACE = AtaxxActionSpace()
