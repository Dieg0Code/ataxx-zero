from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
from typing_extensions import NotRequired

from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2
from game.types import Grid


class PositionCountState(TypedDict):
    grid: list[list[int]]
    current_player: int
    count: int


class BoardState(TypedDict):
    grid: list[list[int]]
    current_player: int
    half_moves: int
    position_counts: NotRequired[list[PositionCountState]]


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
    current_key = board._position_key()
    position_counts = board._position_counts
    if position_counts.get(current_key, 0) <= 0:
        # Some tests and API flows build terminal boards by mutating `grid`
        # directly. If repetition history no longer matches the visible board,
        # serialize a coherent baseline instead of emitting an impossible state.
        position_counts = Counter({current_key: 1})
    return {
        "grid": board.grid.astype(np.int8).tolist(),
        "current_player": int(board.current_player),
        "half_moves": int(board.half_moves),
        "position_counts": [
            {
                "grid": np.frombuffer(position_grid, dtype=np.int8)
                .reshape((BOARD_SIZE, BOARD_SIZE))
                .tolist(),
                "current_player": int(position_player),
                "count": int(count),
            }
            for (position_player, position_grid), count in position_counts.items()
        ],
    }


def _parse_position_counts(
    raw_position_counts: object,
) -> Counter[tuple[int, bytes]] | None:
    if raw_position_counts is None:
        return None
    if not isinstance(raw_position_counts, Sequence) or isinstance(
        raw_position_counts,
        (str, bytes),
    ):
        raise ValueError("position_counts must be a sequence.")

    counts: Counter[tuple[int, bytes]] = Counter()
    for idx, entry in enumerate(raw_position_counts):
        if not isinstance(entry, Mapping):
            raise ValueError(f"position_counts[{idx}] must be a mapping.")
        if "grid" not in entry:
            raise ValueError(f"position_counts[{idx}] missing required key: 'grid'.")
        if "current_player" not in entry:
            raise ValueError(
                f"position_counts[{idx}] missing required key: 'current_player'."
            )
        if "count" not in entry:
            raise ValueError(f"position_counts[{idx}] missing required key: 'count'.")

        grid = _parse_grid(entry["grid"])
        current_player = _ensure_int(
            f"position_counts[{idx}]['current_player']",
            entry["current_player"],
        )
        if current_player not in _VALID_PLAYERS:
            raise ValueError(
                f"position_counts[{idx}]['current_player'] must be one of "
                f"{sorted(_VALID_PLAYERS)}."
            )
        count = _ensure_int(f"position_counts[{idx}]['count']", entry["count"])
        if count <= 0:
            raise ValueError(f"position_counts[{idx}]['count'] must be > 0.")
        counts[(current_player, grid.tobytes())] = count
    return counts


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
    parsed_position_counts = _parse_position_counts(payload.get("position_counts"))
    if parsed_position_counts is not None:
        current_key = board._position_key()
        if current_key not in parsed_position_counts:
            raise ValueError("position_counts must include the current board position.")
        # Full repetition history affects legal terminality, so round-trip it exactly
        # when present instead of silently resetting to the current position only.
        board._position_counts = parsed_position_counts
    return board
