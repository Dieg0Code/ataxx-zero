from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .constants import (
    BOARD_SIZE,
    DRAW,
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    WIN_P1,
    WIN_P2,
)

Grid = npt.NDArray[np.int8]
Move = tuple[int, int, int, int]


class AtaxxBoard:
    """State and rules for Ataxx."""

    def __init__(self, grid: Grid | None = None, player: int = PLAYER_1) -> None:
        self.grid: Grid
        if grid is None:
            self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            self._init_pieces()
        else:
            grid_int8 = np.asarray(grid, dtype=np.int8)
            if grid_int8.shape != (BOARD_SIZE, BOARD_SIZE):
                raise ValueError(
                    f"grid must have shape {(BOARD_SIZE, BOARD_SIZE)}, got {grid_int8.shape}"
                )
            self.grid = grid_int8

        self.current_player: int = player
        # Variant anti-loop rule: hard cap on total half-moves.
        self.half_moves = 0

    def _init_pieces(self) -> None:
        """Standard opening with opposite corners occupied."""
        self.grid[0, 0] = PLAYER_1
        self.grid[BOARD_SIZE - 1, BOARD_SIZE - 1] = PLAYER_1
        self.grid[0, BOARD_SIZE - 1] = PLAYER_2
        self.grid[BOARD_SIZE - 1, 0] = PLAYER_2

    def copy(self) -> AtaxxBoard:
        new_board = AtaxxBoard(self.grid.copy(), self.current_player)
        new_board.half_moves = self.half_moves
        return new_board

    @staticmethod
    def _opponent(player: int) -> int:
        return -player

    def _has_move_for(self, player: int) -> bool:
        piece_coords = np.argwhere(self.grid == player)
        for r, c in piece_coords:
            r_min, r_max = max(0, r - 2), min(BOARD_SIZE, r + 3)
            c_min, c_max = max(0, c - 2), min(BOARD_SIZE, c + 3)
            if np.any(self.grid[r_min:r_max, c_min:c_max] == EMPTY):
                return True
        return False

    def get_valid_moves(self, player: int | None = None) -> list[Move]:
        """Generate all legal moves for a player."""
        p = self.current_player if player is None else player
        moves: list[Move] = []
        piece_coords = np.argwhere(self.grid == p)

        for r, c in piece_coords:
            r_min = max(0, r - 2)
            r_max = min(BOARD_SIZE, r + 3)
            c_min = max(0, c - 2)
            c_max = min(BOARD_SIZE, c + 3)

            for tr in range(r_min, r_max):
                for tc in range(c_min, c_max):
                    if self.grid[tr, tc] == EMPTY and (r != tr or c != tc):
                        moves.append((int(r), int(c), tr, tc))

        return moves

    def has_valid_moves(self, player: int | None = None) -> bool:
        """Fast check for at least one legal move."""
        p = self.current_player if player is None else player
        return self._has_move_for(p)

    def step(self, move: Move | None) -> None:
        """
        Apply one move to the current state.

        `move=None` is treated as a pass and is only legal when there are no moves.
        """
        if move is None:
            if self.has_valid_moves():
                raise ValueError("Pass is illegal when legal moves exist.")
            self.current_player = self._opponent(self.current_player)
            self.half_moves += 1
            return

        r_start, c_start, r_end, c_end = move

        if self.grid[r_start, c_start] != self.current_player:
            raise ValueError(
                f"Cannot move a non-current piece from ({r_start}, {c_start})."
            )
        if self.grid[r_end, c_end] != EMPTY:
            raise ValueError(f"Destination ({r_end}, {c_end}) is not empty.")

        dist = max(abs(r_start - r_end), abs(c_start - c_end))
        if dist == 1:
            self.grid[r_end, c_end] = self.current_player
        elif dist == 2:
            self.grid[r_end, c_end] = self.current_player
            self.grid[r_start, c_start] = EMPTY
        else:
            raise ValueError(f"Illegal move distance: {dist}.")

        self._infect_neighbors(r_end, c_end)
        self.current_player = self._opponent(self.current_player)
        self.half_moves += 1

    def _infect_neighbors(self, r: int, c: int) -> None:
        """Convert adjacent opponent pieces around (r, c)."""
        opponent = self._opponent(self.current_player)
        r_min = max(0, r - 1)
        r_max = min(BOARD_SIZE, r + 2)
        c_min = max(0, c - 1)
        c_max = min(BOARD_SIZE, c + 2)
        window = self.grid[r_min:r_max, c_min:c_max]
        window[window == opponent] = self.current_player

    def is_game_over(self) -> bool:
        """
        End conditions:
        1) board is full,
        2) one side has no pieces,
        3) half-move cap reached (variant anti-loop rule),
        4) both players have no legal moves.
        """
        if not np.any(self.grid == EMPTY):
            return True

        if not np.any(self.grid == PLAYER_1) or not np.any(self.grid == PLAYER_2):
            return True

        if self.half_moves >= 100:
            return True

        return not self._has_move_for(self.current_player) and not self._has_move_for(
            self._opponent(self.current_player)
        )

    def get_result(self) -> int:
        """Return game result from PLAYER_1 perspective."""
        if not self.is_game_over():
            raise ValueError("Result is only defined when the game is over.")

        p1_count = int(np.sum(self.grid == PLAYER_1))
        p2_count = int(np.sum(self.grid == PLAYER_2))

        if p1_count == 0:
            return WIN_P2
        if p2_count == 0:
            return WIN_P1

        if p1_count > p2_count:
            return WIN_P1
        if p2_count > p1_count:
            return WIN_P2
        return DRAW

    def get_canonical_form(self) -> np.ndarray:
        """
        Current-player perspective:
        own=+1, opponent=-1, empty=0.
        """
        if self.current_player == PLAYER_1:
            return self.grid.copy()
        return -self.grid

    def get_observation(self) -> np.ndarray:
        """
        3-channel observation for NN:
        0: own pieces, 1: opponent pieces, 2: empty squares.
        """
        obs = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = np.asarray(self.grid == self.current_player, dtype=np.float32)
        obs[1] = np.asarray(
            self.grid == self._opponent(self.current_player),
            dtype=np.float32,
        )
        obs[2] = np.asarray(self.grid == EMPTY, dtype=np.float32)
        return obs

    def __str__(self) -> str:
        mapping = {PLAYER_1: "X", PLAYER_2: "O", EMPTY: "."}
        lines = []
        for row in range(BOARD_SIZE):
            line = " ".join(
                mapping[int(self.grid[row, col])] for col in range(BOARD_SIZE)
            )
            lines.append(line)
        return "\n".join(lines)
