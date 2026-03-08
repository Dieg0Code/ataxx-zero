from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from game.constants import BOARD_SIZE
from game.rules import opponent
from training.config_runtime import cfg_bool, cfg_float

if TYPE_CHECKING:
    from data.replay_buffer import TrainingExample
    from game.board import AtaxxBoard

PolicyArray: TypeAlias = np.ndarray
ObservationHistoryEntry: TypeAlias = tuple[np.ndarray, PolicyArray, int]
ShapedHistoryEntry: TypeAlias = tuple[np.ndarray, PolicyArray, int, float]
HistoryEntry: TypeAlias = ObservationHistoryEntry | ShapedHistoryEntry


def compute_state_potential(board: AtaxxBoard, player: int) -> float:
    own_count = float(board.p1_count if player == 1 else board.p2_count)
    opp_count = float(board.p2_count if player == 1 else board.p1_count)
    piece_balance = (own_count - opp_count) / float(BOARD_SIZE * BOARD_SIZE)

    own_moves = len(board.get_valid_moves(player))
    opp_moves = len(board.get_valid_moves(opponent(player)))
    mobility_den = max(1.0, float(own_moves + opp_moves))
    mobility_balance = float(own_moves - opp_moves) / mobility_den

    material_weight = max(0.0, cfg_float("reward_shaping_material_weight"))
    mobility_weight = max(0.0, cfg_float("reward_shaping_mobility_weight"))
    total_weight = material_weight + mobility_weight
    if total_weight <= 0.0:
        material_weight, mobility_weight = 1.0, 0.0
    else:
        material_weight /= total_weight
        mobility_weight /= total_weight

    potential = (material_weight * piece_balance) + (mobility_weight * mobility_balance)
    return float(np.clip(potential, -1.0, 1.0))


def compute_transition_shaping_reward(
    *,
    before_potential: float,
    after_potential: float,
) -> float:
    gamma = cfg_float("reward_shaping_gamma")
    return float((gamma * after_potential) - before_potential)


def outcome_value_target(
    *,
    player_at_turn: int,
    winner: int,
    forced_draw: bool,
) -> float:
    if winner == 0:
        return -cfg_float("reward_shaping_draw_penalty") if forced_draw else 0.0
    return 1.0 if winner == player_at_turn else -1.0


def _normalize_history_entry(entry: HistoryEntry) -> ShapedHistoryEntry:
    if len(entry) == 4:
        observation, policy, player_at_turn, shaping_reward = entry
        return observation, policy, player_at_turn, float(shaping_reward)
    observation, policy, player_at_turn = entry
    return observation, policy, player_at_turn, 0.0


def history_to_examples(
    game_history: Sequence[HistoryEntry],
    *,
    winner: int,
    forced_draw: bool = False,
) -> list[TrainingExample]:
    if len(game_history) == 0:
        return []

    examples_reversed: list[TrainingExample] = []
    future_shaping = 0.0
    next_player: int | None = None
    shaping_scale = cfg_float("reward_shaping_scale") if cfg_bool("reward_shaping_enabled") else 0.0
    gamma = cfg_float("reward_shaping_gamma")

    # Stored turns may belong to the same side repeatedly (model vs heuristic) or alternate
    # (self-play). We preserve perspective by only flipping the future suffix when control
    # passes to the other side between stored decisions.
    for entry in reversed(game_history):
        observation, policy, player_at_turn, shaping_reward = _normalize_history_entry(entry)
        if next_player is None or next_player == player_at_turn:
            future_shaping = shaping_reward + (gamma * future_shaping)
        else:
            future_shaping = shaping_reward - (gamma * future_shaping)
        next_player = player_at_turn

        z = outcome_value_target(
            player_at_turn=player_at_turn,
            winner=winner,
            forced_draw=forced_draw,
        )
        target = float(np.clip(z + (shaping_scale * future_shaping), -1.0, 1.0))
        examples_reversed.append((observation, policy, target))

    examples_reversed.reverse()
    return examples_reversed


__all__ = [
    "HistoryEntry",
    "ShapedHistoryEntry",
    "compute_state_potential",
    "compute_transition_shaping_reward",
    "history_to_examples",
    "outcome_value_target",
]
