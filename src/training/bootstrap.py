from __future__ import annotations

from typing import Literal

import numpy as np

from agents.heuristic import heuristic_move
from data.replay_buffer import TrainingExample
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard

HeuristicLevel = Literal["easy", "normal", "hard"]
HistoryEntry = tuple[np.ndarray, np.ndarray, int]


def _one_hot_policy(action_idx: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
    policy[action_idx] = 1.0
    return policy


def history_to_examples(
    game_history: list[HistoryEntry],
    winner: int,
) -> list[TrainingExample]:
    """Convert per-turn history into value targets from the acting player's perspective."""
    examples: list[TrainingExample] = []
    for observation, policy, player_at_turn in game_history:
        if winner == 0:
            z = 0.0
        elif winner == player_at_turn:
            z = 1.0
        else:
            z = -1.0
        examples.append((observation, policy, z))
    return examples


def generate_imitation_data(
    *,
    n_games: int,
    heuristic_level: HeuristicLevel = "hard",
    seed: int = 42,
) -> list[TrainingExample]:
    """
    Generate supervised warmup data from heuristic-vs-heuristic games.

    Pedagogical intent:
    this gives the policy head legal/tactical priors before self-play RL,
    reducing cold-start collapse into repetitive draw loops.
    """
    if n_games <= 0:
        return []

    rng = np.random.default_rng(seed=seed)
    all_examples: list[TrainingExample] = []

    for _ in range(n_games):
        board = AtaxxBoard()
        game_history: list[HistoryEntry] = []

        while not board.is_game_over():
            player_at_turn = int(board.current_player)
            move = heuristic_move(board=board, rng=rng, level=heuristic_level)
            action_idx = ACTION_SPACE.encode(move)
            policy = _one_hot_policy(action_idx)
            game_history.append((board.get_observation(), policy, player_at_turn))
            board.step(move)

        winner = board.get_result()
        all_examples.extend(history_to_examples(game_history=game_history, winner=winner))

    return all_examples
