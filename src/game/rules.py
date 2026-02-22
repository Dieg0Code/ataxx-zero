from __future__ import annotations

from game.types import Player


def opponent(player: Player) -> Player:
    return -player


def move_distance(r_start: int, c_start: int, r_end: int, c_end: int) -> int:
    return max(abs(r_start - r_end), abs(c_start - c_end))


def is_clone_move(distance: int) -> bool:
    return distance == 1


def is_jump_move(distance: int) -> bool:
    return distance == 2

