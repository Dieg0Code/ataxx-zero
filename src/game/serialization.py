from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np

from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2
from game.types import Grid


class BoardState(TypedDict):
    grid: list[list[int]]
    current_player: int
    half_moves: int


_VALID_CELL_VALUES = {EMPTY, PLAYER_1, PLAYER_2}
_VALID_PLAYERS = {PLAYER_1, PLAYER_2}


def _ensure_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _parse_grid(grid_raw: object) -> Grid:
    if not isinstance(grid_raw, Sequence) or isinstance(grid_raw, (str, bytes)):
        raise ValueError("grid must be a 2D sequence.")
    if len(grid_raw) != BOARD_SIZE:
        raise ValueError(f"grid must have {BOARD_SIZE} rows.")

    rows: list[list[int]] = []
    for r_idx, row in enumerate(grid_raw):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"grid row {r_idx} must be a sequence.")
        if len(row) != BOARD_SIZE:
            raise ValueError(f"grid row {r_idx} must have {BOARD_SIZE} columns.")

        parsed_row: list[int] = []
        for c_idx, cell in enumerate(row):
            cell_int = _ensure_int(f"grid[{r_idx}][{c_idx}]", cell)
            if cell_int not in _VALID_CELL_VALUES:
                raise ValueError(
                    f"grid[{r_idx}][{c_idx}] must be one of {sorted(_VALID_CELL_VALUES)}."
                )
            parsed_row.append(cell_int)
        rows.append(parsed_row)

    return np.asarray(rows, dtype=np.int8)


def board_to_state(board: AtaxxBoard) -> BoardState:
    return {
        "grid": board.grid.astype(np.int8).tolist(),
        "current_player": int(board.current_player),
        "half_moves": int(board.half_moves),
    }


def board_from_state(payload: Mapping[str, object]) -> AtaxxBoard:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping with board fields.")

    if "grid" not in payload:
        raise ValueError("payload missing required key: 'grid'.")
    if "current_player" not in payload:
        raise ValueError("payload missing required key: 'current_player'.")
    if "half_moves" not in payload:
        raise ValueError("payload missing required key: 'half_moves'.")

    grid = _parse_grid(payload["grid"])
    current_player = _ensure_int("current_player", payload["current_player"])
    if current_player not in _VALID_PLAYERS:
        raise ValueError(f"current_player must be one of {sorted(_VALID_PLAYERS)}.")

    half_moves = _ensure_int("half_moves", payload["half_moves"])
    if half_moves < 0:
        raise ValueError("half_moves must be >= 0.")

    board = AtaxxBoard(grid=grid, player=current_player)
    board.half_moves = half_moves
    return board
