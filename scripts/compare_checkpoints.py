from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


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


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()

    from inference.checkpoint_duel_runtime import (
        play_checkpoint_match_results,
        summarize_match_results,
    )

    checkpoint_a = Path(args.checkpoint_a)
    checkpoint_b = Path(args.checkpoint_b)
    if not checkpoint_a.exists():
        raise FileNotFoundError(f"Checkpoint A not found: {checkpoint_a}")
    if not checkpoint_b.exists():
        raise FileNotFoundError(f"Checkpoint B not found: {checkpoint_b}")

    device = _resolve_device(args.device)
    results = play_checkpoint_match_results(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        games=max(1, int(args.games)),
        device=device,
        mcts_sims=int(args.mcts_sims),
        c_puct=float(args.c_puct),
        seed=int(args.seed),
    )

    for idx, result in enumerate(results, start=1):
        color_a = "p1" if int(result["checkpoint_a_player"]) == 1 else "p2"
        print(
            f"[{idx}/{len(results)}] "
            f"checkpoint_a={color_a} winner={int(result['winner'])} turns={int(result['turns'])}",
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
