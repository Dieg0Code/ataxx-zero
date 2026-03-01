from __future__ import annotations

from collections import Counter

import numpy as np

from .constants import (
    BOARD_SIZE,
    DRAW,
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    WIN_P1,
    WIN_P2,
)
from .rules import is_clone_move, is_jump_move, move_distance, opponent
from .types import Grid, Move, Player


def _build_radius2_targets() -> tuple[tuple[tuple[int, int], ...], ...]:
    targets: list[tuple[tuple[int, int], ...]] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell_targets: list[tuple[int, int]] = []
            r_min = max(0, r - 2)
            r_max = min(BOARD_SIZE, r + 3)
            c_min = max(0, c - 2)
            c_max = min(BOARD_SIZE, c + 3)
            for tr in range(r_min, r_max):
                for tc in range(c_min, c_max):
                    if tr == r and tc == c:
                        continue
                    cell_targets.append((tr, tc))
            targets.append(tuple(cell_targets))
    return tuple(targets)


_RADIUS2_TARGETS = _build_radius2_targets()


class AtaxxBoard:
    """State and rules for Ataxx."""

    def __init__(self, grid: Grid | None = None, player: Player = PLAYER_1) -> None:
        self.grid: Grid
        self.p1_count: int
        self.p2_count: int
        self.empty_count: int
        if grid is None:
            self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            self._init_pieces()
            self.p1_count = 2
            self.p2_count = 2
            self.empty_count = BOARD_SIZE * BOARD_SIZE - 4
        else:
            grid_int8 = np.asarray(grid, dtype=np.int8)
            if grid_int8.shape != (BOARD_SIZE, BOARD_SIZE):
                raise ValueError(
                    f"grid must have shape {(BOARD_SIZE, BOARD_SIZE)}, got {grid_int8.shape}"
                )
            self.grid = grid_int8
            self.p1_count = int(np.sum(self.grid == PLAYER_1))
            self.p2_count = int(np.sum(self.grid == PLAYER_2))
            self.empty_count = int(np.sum(self.grid == EMPTY))

        self.current_player: Player = player
        # Variant anti-loop rule: hard cap on total half-moves.
        self.half_moves = 0
        # Track repeated positions including side-to-move.
        self._position_counts: Counter[tuple[int, bytes]] = Counter()
        self._position_counts[self._position_key()] = 1

    def _init_pieces(self) -> None:
        """Standard opening with opposite corners occupied."""
        self.grid[0, 0] = PLAYER_1
        self.grid[BOARD_SIZE - 1, BOARD_SIZE - 1] = PLAYER_1
        self.grid[0, BOARD_SIZE - 1] = PLAYER_2
        self.grid[BOARD_SIZE - 1, 0] = PLAYER_2

    def copy(self) -> AtaxxBoard:
        # Fast internal clone used heavily by MCTS rollouts.
        # We can bypass __init__ checks because current state is already valid.
        new_board = object.__new__(AtaxxBoard)
        new_board.grid = self.grid.copy()
        new_board.current_player = self.current_player
        new_board.half_moves = self.half_moves
        new_board.p1_count = self.p1_count
        new_board.p2_count = self.p2_count
        new_board.empty_count = self.empty_count
        new_board._position_counts = Counter(self._position_counts)
        return new_board

    def copy_from(self, other: AtaxxBoard) -> None:
        """In-place state copy for hot loops (e.g., MCTS simulations)."""
        np.copyto(self.grid, other.grid)
        self.current_player = other.current_player
        self.half_moves = other.half_moves
        self.p1_count = other.p1_count
        self.p2_count = other.p2_count
        self.empty_count = other.empty_count
        self._position_counts = Counter(other._position_counts)

    def _position_key(self) -> tuple[int, bytes]:
        return int(self.current_player), self.grid.tobytes()

    def _has_move_for(self, player: Player) -> bool:
        piece_coords = np.argwhere(self.grid == player)
        for r, c in piece_coords:
            rr = int(r)
            cc = int(c)
            origin_idx = rr * BOARD_SIZE + cc
            for tr, tc in _RADIUS2_TARGETS[origin_idx]:
                if self.grid[tr, tc] == EMPTY:
                    return True
        return False

    def get_valid_moves(self, player: Player | None = None) -> list[Move]:
        """Generate all legal moves for a player."""
        p = self.current_player if player is None else player
        moves: list[Move] = []
        piece_coords = np.argwhere(self.grid == p)

        for r, c in piece_coords:
            rr = int(r)
            cc = int(c)
            origin_idx = rr * BOARD_SIZE + cc
            for tr, tc in _RADIUS2_TARGETS[origin_idx]:
                if self.grid[tr, tc] == EMPTY:
                    moves.append((rr, cc, tr, tc))

        return moves

    def has_valid_moves(self, player: Player | None = None) -> bool:
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
            self.current_player = opponent(self.current_player)
            self.half_moves += 1
            self._position_counts[self._position_key()] += 1
            return

        r_start, c_start, r_end, c_end = move

        if self.grid[r_start, c_start] != self.current_player:
            raise ValueError(
                f"Cannot move a non-current piece from ({r_start}, {c_start})."
            )
        if self.grid[r_end, c_end] != EMPTY:
            raise ValueError(f"Destination ({r_end}, {c_end}) is not empty.")

        dist = move_distance(r_start, c_start, r_end, c_end)
        if is_clone_move(dist):
            self.grid[r_end, c_end] = self.current_player
            self.empty_count -= 1
            if self.current_player == PLAYER_1:
                self.p1_count += 1
            else:
                self.p2_count += 1
        elif is_jump_move(dist):
            self.grid[r_end, c_end] = self.current_player
            self.grid[r_start, c_start] = EMPTY
        else:
            raise ValueError(f"Illegal move distance: {dist}.")

        self._infect_neighbors(r_end, c_end)
        self.current_player = opponent(self.current_player)
        self.half_moves += 1
        self._position_counts[self._position_key()] += 1

    def _infect_neighbors(self, r: int, c: int) -> None:
        """Convert adjacent opponent pieces around (r, c)."""
        enemy = opponent(self.current_player)
        r_min = max(0, r - 1)
        r_max = min(BOARD_SIZE, r + 2)
        c_min = max(0, c - 1)
        c_max = min(BOARD_SIZE, c + 2)
        window = self.grid[r_min:r_max, c_min:c_max]
        converted = int(np.sum(window == enemy))
        window[window == enemy] = self.current_player
        if converted > 0:
            if self.current_player == PLAYER_1:
                self.p1_count += converted
                self.p2_count -= converted
            else:
                self.p2_count += converted
                self.p1_count -= converted

    def is_game_over(self) -> bool:
        """
        End conditions:
        1) board is full,
        2) one side has no pieces,
        3) half-move cap reached (variant anti-loop rule),
        4) both players have no legal moves.
        """
        if self.empty_count == 0:
            return True

        if self.p1_count == 0 or self.p2_count == 0:
            return True

        if self.half_moves >= 100:
            return True

        if max(self._position_counts.values(), default=0) >= 3:
            return True

        return not self._has_move_for(self.current_player) and not self._has_move_for(
            opponent(self.current_player)
        )

    def get_result(self) -> int:
        """Return game result from PLAYER_1 perspective."""
        if not self.is_game_over():
            raise ValueError("Result is only defined when the game is over.")

        # Threefold repetition is a forced draw regardless of piece counts.
        if self._position_counts[self._position_key()] >= 3:
            return DRAW

        p1_count = self.p1_count
        p2_count = self.p2_count

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
        4-channel observation for NN:
        0: own pieces, 1: opponent pieces, 2: empty squares, 3: half-move progress.
        """
        obs = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = np.asarray(self.grid == self.current_player, dtype=np.float32)
        obs[1] = np.asarray(
            self.grid == opponent(self.current_player),
            dtype=np.float32,
        )
        obs[2] = np.asarray(self.grid == EMPTY, dtype=np.float32)
        obs[3] = np.full(
            (BOARD_SIZE, BOARD_SIZE),
            min(1.0, float(self.half_moves) / 100.0),
            dtype=np.float32,
        )
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
