from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
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

    if checkpoint_path.suffix == ".ckpt":
        return AtaxxZero.load_from_checkpoint(str(checkpoint_path), map_location=device)

    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint format: expected dictionary.")
    state_dict_obj = payload.get("state_dict")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("Checkpoint dictionary must contain key 'state_dict'.")

    hparams = payload.get("hparams")
    kwargs: dict[str, Any] = {}
    if isinstance(hparams, dict):
        allowed = {"d_model", "nhead", "num_layers", "dim_feedforward", "dropout"}
        kwargs = {key: hparams[key] for key in allowed if key in hparams}

    system = AtaxxZero(**kwargs)
    system.load_state_dict(state_dict_obj)
    system.eval()
    system.to(device)
    return system


__all__ = [
    "build_match_schedule",
    "load_system_from_checkpoint",
    "summarize_match_results",
]
