from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from model.checkpoint_compat import (
    adapt_state_dict_observation_channels,
    extract_checkpoint_state_dict,
    extract_model_kwargs,
)

if TYPE_CHECKING:
    from engine.mcts import MCTS
    from game.board import AtaxxBoard
    from model.system import AtaxxZero

MatchSchedule = list[tuple[int, int]]
MatchResult = dict[str, int]


def build_match_schedule(*, games: int) -> MatchSchedule:
    if games <= 0:
        return []
    schedule: MatchSchedule = []
    for idx in range(games):
        checkpoint_a_player = 1 if idx % 2 == 0 else -1
        checkpoint_b_player = -checkpoint_a_player
        schedule.append((checkpoint_a_player, checkpoint_b_player))
    return schedule


def summarize_match_results(*, results: list[MatchResult]) -> dict[str, float | int]:
    games = len(results)
    if games == 0:
        return {
            "games": 0,
            "checkpoint_a_wins": 0,
            "checkpoint_b_wins": 0,
            "draws": 0,
            "checkpoint_a_score": 0.0,
            "avg_turns": 0.0,
        }

    checkpoint_a_wins = 0
    checkpoint_b_wins = 0
    draws = 0
    total_turns = 0
    for result in results:
        winner = int(result["winner"])
        checkpoint_a_player = int(result["checkpoint_a_player"])
        total_turns += int(result["turns"])
        if winner == 0:
            draws += 1
        elif winner == checkpoint_a_player:
            checkpoint_a_wins += 1
        else:
            checkpoint_b_wins += 1

    checkpoint_a_score = (checkpoint_a_wins + (0.5 * draws)) / float(games)
    return {
        "games": games,
        "checkpoint_a_wins": checkpoint_a_wins,
        "checkpoint_b_wins": checkpoint_b_wins,
        "draws": draws,
        "checkpoint_a_score": checkpoint_a_score,
        "avg_turns": total_turns / float(games),
    }


def load_system_from_checkpoint(checkpoint_path: Path, *, device: str) -> AtaxxZero:
    from model.system import AtaxxZero

    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint format: expected dictionary.")
    state_dict_obj = extract_checkpoint_state_dict(payload)
    system = AtaxxZero(**extract_model_kwargs(payload))
    system.load_state_dict(
        adapt_state_dict_observation_channels(
            state_dict_obj,
            target_channels=int(system.model.num_input_channels),
        )
    )
    system.eval()
    system.to(device)
    return system


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _pick_model_action_idx(board: AtaxxBoard, mcts: MCTS) -> int:
    probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
    return int(np.argmax(probs))


def play_checkpoint_match_results(
    *,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    games: int,
    device: str,
    mcts_sims: int,
    c_puct: float,
    seed: int,
) -> list[MatchResult]:
    from engine.mcts import MCTS
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard

    checkpoint_a_path = Path(checkpoint_a)
    checkpoint_b_path = Path(checkpoint_b)
    if not checkpoint_a_path.exists():
        raise FileNotFoundError(f"Checkpoint A not found: {checkpoint_a_path}")
    if not checkpoint_b_path.exists():
        raise FileNotFoundError(f"Checkpoint B not found: {checkpoint_b_path}")

    resolved_device = resolve_device(device)
    system_a = load_system_from_checkpoint(checkpoint_a_path, device=resolved_device)
    system_b = load_system_from_checkpoint(checkpoint_b_path, device=resolved_device)
    mcts_a = MCTS(
        model=system_a.model,
        c_puct=c_puct,
        n_simulations=mcts_sims,
        device=resolved_device,
    )
    mcts_b = MCTS(
        model=system_b.model,
        c_puct=c_puct,
        n_simulations=mcts_sims,
        device=resolved_device,
    )

    schedule = build_match_schedule(games=max(1, int(games)))
    rng = np.random.default_rng(seed=int(seed))
    results: list[MatchResult] = []

    for checkpoint_a_player, checkpoint_b_player in schedule:
        board = AtaxxBoard()
        turn_seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(turn_seed)
        np.random.seed(turn_seed)
        turns = 0
        while not board.is_game_over():
            turns += 1
            if board.current_player == checkpoint_a_player:
                action_idx = _pick_model_action_idx(board, mcts_a)
            elif board.current_player == checkpoint_b_player:
                action_idx = _pick_model_action_idx(board, mcts_b)
            else:
                raise RuntimeError("Unexpected player assignment while comparing checkpoints.")
            board.step(ACTION_SPACE.decode(action_idx))

        results.append(
            {
                "winner": int(board.get_result()),
                "turns": turns,
                "checkpoint_a_player": checkpoint_a_player,
            },
        )

    return results


def run_match_results_to_summary(
    *,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    games: int,
    device: str,
    mcts_sims: int,
    c_puct: float,
    seed: int,
) -> dict[str, float | int]:
    results = play_checkpoint_match_results(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        games=games,
        device=device,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
        seed=seed,
    )
    return summarize_match_results(results=results)


def play_checkpoint_vs_heuristic_results(
    *,
    checkpoint: str | Path,
    heuristic_level: str,
    games: int,
    device: str,
    mcts_sims: int,
    c_puct: float,
    seed: int,
) -> list[dict[str, int]]:
    from agents.heuristic import heuristic_move
    from engine.mcts import MCTS
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = resolve_device(device)
    system = load_system_from_checkpoint(checkpoint_path, device=resolved_device)
    mcts = MCTS(
        model=system.model,
        c_puct=c_puct,
        n_simulations=mcts_sims,
        device=resolved_device,
    )

    rng = np.random.default_rng(seed=int(seed))
    results: list[dict[str, int]] = []
    for _ in range(max(1, int(games))):
        board = AtaxxBoard()
        model_player = 1 if float(rng.random()) >= 0.5 else -1
        root = None
        turns = 0
        while not board.is_game_over():
            turns += 1
            if board.current_player == model_player:
                probs, root = mcts.run_with_root(
                    board=board,
                    root=root,
                    add_dirichlet_noise=False,
                    temperature=0.0,
                )
                action_idx = int(np.argmax(probs))
                board.step(ACTION_SPACE.decode(action_idx))
                root = mcts.advance_root(root, action_idx)
                continue
            move = heuristic_move(board, rng, heuristic_level)
            board.step(move)
            root = mcts.advance_root(root, ACTION_SPACE.encode(move))

        winner = int(board.get_result())
        checkpoint_score = 1 if winner == model_player else (0 if winner == 0 else -1)
        results.append(
            {
                "winner": checkpoint_score,
                "turns": turns,
                "checkpoint_a_player": 1,
            },
        )

    return results


def run_checkpoint_vs_heuristic_to_summary(
    *,
    checkpoint: str | Path,
    heuristic_level: str,
    games: int,
    device: str,
    mcts_sims: int,
    c_puct: float,
    seed: int,
) -> dict[str, float | int]:
    results = play_checkpoint_vs_heuristic_results(
        checkpoint=checkpoint,
        heuristic_level=heuristic_level,
        games=games,
        device=device,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
        seed=seed,
    )
    return summarize_match_results(results=results)


__all__ = [
    "build_match_schedule",
    "load_system_from_checkpoint",
    "play_checkpoint_match_results",
    "play_checkpoint_vs_heuristic_results",
    "resolve_device",
    "run_checkpoint_vs_heuristic_to_summary",
    "run_match_results_to_summary",
    "summarize_match_results",
]
