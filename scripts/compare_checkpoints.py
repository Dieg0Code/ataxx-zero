from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from engine.mcts import MCTS
    from game.board import AtaxxBoard


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short automated duel between two Ataxx checkpoints.",
    )
    parser.add_argument("--checkpoint-a", required=True, help="Path to checkpoint A (.pt/.ckpt).")
    parser.add_argument("--checkpoint-b", required=True, help="Path to checkpoint B (.pt/.ckpt).")
    parser.add_argument("--games", type=int, default=8, help="Number of games to play.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mcts-sims", "--sims", type=int, default=96)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _pick_model_action_idx(board: AtaxxBoard, mcts: MCTS) -> int:
    probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
    return int(np.argmax(probs))


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()

    from engine.mcts import MCTS
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard
    from inference.checkpoint_duel_runtime import (
        build_match_schedule,
        load_system_from_checkpoint,
        summarize_match_results,
    )

    checkpoint_a = Path(args.checkpoint_a)
    checkpoint_b = Path(args.checkpoint_b)
    if not checkpoint_a.exists():
        raise FileNotFoundError(f"Checkpoint A not found: {checkpoint_a}")
    if not checkpoint_b.exists():
        raise FileNotFoundError(f"Checkpoint B not found: {checkpoint_b}")

    device = _resolve_device(args.device)
    system_a = load_system_from_checkpoint(checkpoint_a, device=device)
    system_b = load_system_from_checkpoint(checkpoint_b, device=device)
    mcts_a = MCTS(model=system_a.model, c_puct=args.c_puct, n_simulations=args.mcts_sims, device=device)
    mcts_b = MCTS(model=system_b.model, c_puct=args.c_puct, n_simulations=args.mcts_sims, device=device)

    schedule = build_match_schedule(games=max(1, int(args.games)))
    rng = np.random.default_rng(seed=int(args.seed))
    results: list[dict[str, int]] = []

    for idx, (checkpoint_a_player, checkpoint_b_player) in enumerate(schedule, start=1):
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

        winner = board.get_result()
        results.append(
            {
                "winner": int(winner),
                "turns": turns,
                "checkpoint_a_player": checkpoint_a_player,
            },
        )
        color_a = "p1" if checkpoint_a_player == 1 else "p2"
        print(
            f"[{idx}/{len(schedule)}] "
            f"checkpoint_a={color_a} winner={winner} turns={turns}",
        )

    summary = summarize_match_results(results=results)
    output: dict[str, float | int | str] = {
        **summary,
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "device": device,
        "mcts_sims": int(args.mcts_sims),
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print("")
    print("Summary")
    print(f"  checkpoint_a: {checkpoint_a}")
    print(f"  checkpoint_b: {checkpoint_b}")
    print(f"  games: {summary['games']}")
    print(f"  checkpoint_a_wins: {summary['checkpoint_a_wins']}")
    print(f"  checkpoint_b_wins: {summary['checkpoint_b_wins']}")
    print(f"  draws: {summary['draws']}")
    print(f"  checkpoint_a_score: {float(summary['checkpoint_a_score']):.3f}")
    print(f"  avg_turns: {float(summary['avg_turns']):.1f}")


if __name__ == "__main__":
    main()
