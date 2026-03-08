from __future__ import annotations

from typing import Literal

import numpy as np

from agents.heuristic import heuristic_move, is_supported_heuristic_level
from data.replay_buffer import TrainingExample
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from training.config_runtime import cfg_bool
from training.reward_runtime import (
    HistoryEntry,
    compute_state_potential,
    compute_transition_shaping_reward,
    history_to_examples,
)

HeuristicLevel = Literal["easy", "normal", "hard", "apex", "gambit", "sentinel"]


def _one_hot_policy(action_idx: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
    policy[action_idx] = 1.0
    return policy


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
    if not is_supported_heuristic_level(heuristic_level):
        raise ValueError(f"Unsupported heuristic level for warmup: {heuristic_level}")

    rng = np.random.default_rng(seed=seed)
    all_examples: list[TrainingExample] = []

    for _ in range(n_games):
        board = AtaxxBoard()
        game_history: list[HistoryEntry] = []
        shaping_enabled = cfg_bool("reward_shaping_enabled")

        while not board.is_game_over():
            player_at_turn = int(board.current_player)
            observation = board.get_observation()
            move = heuristic_move(board=board, rng=rng, level=heuristic_level)
            action_idx = ACTION_SPACE.encode(move)
            policy = _one_hot_policy(action_idx)
            shaping_reward = 0.0
            before_potential = 0.0
            if shaping_enabled:
                before_potential = compute_state_potential(board, player_at_turn)
            board.step(move)
            if shaping_enabled:
                after_potential = compute_state_potential(board, player_at_turn)
                shaping_reward = compute_transition_shaping_reward(
                    before_potential=before_potential,
                    after_potential=after_potential,
                )
            game_history.append((observation, policy, player_at_turn, shaping_reward))

        winner = board.get_result()
        all_examples.extend(
            history_to_examples(
                game_history=game_history,
                winner=winner,
                forced_draw=board.is_forced_draw(),
            ),
        )

    return all_examples
