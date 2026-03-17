from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint duel and update a local Elo league for training evaluation.",
    )
    parser.add_argument("--league-path", default="checkpoints/league_ratings.json")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", default=None)
    parser.add_argument("--participant-a-id", default=None)
    parser.add_argument("--participant-b-id", default=None)
    parser.add_argument("--participant-a-name", default=None)
    parser.add_argument("--participant-b-name", default=None)
    parser.add_argument(
        "--heuristic-levels",
        default="",
        help="Comma-separated heuristic anchors to evaluate and add to league (e.g. hard,apex,sentinel).",
    )
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument(
        "--anchor-games",
        type=int,
        default=None,
        help="Games per heuristic anchor. Defaults to --games.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mcts-sims", "--sims", type=int, default=96)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _default_participant_id(prefix: str, checkpoint: Path) -> str:
    return f"{prefix}:{checkpoint.stem}"


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()

    from agents.heuristic import HEURISTIC_LEVEL_SET
    from inference.checkpoint_duel_runtime import (
        resolve_device,
        run_checkpoint_vs_heuristic_to_summary,
        run_match_results_to_summary,
    )
    from inference.checkpoint_league_runtime import (
        apply_series_to_league,
        choose_champion_id,
        load_league_state,
        save_league_state,
    )

    checkpoint_a = Path(args.checkpoint_a)
    checkpoint_b = Path(args.checkpoint_b) if args.checkpoint_b is not None else None
    participant_a_id = args.participant_a_id or _default_participant_id("ckpt", checkpoint_a)
    participant_a_name = args.participant_a_name or checkpoint_a.stem
    resolved_device = resolve_device(args.device)
    heuristic_levels = [level.strip() for level in args.heuristic_levels.split(",") if level.strip() != ""]
    for level in heuristic_levels:
        if level not in HEURISTIC_LEVEL_SET:
            raise ValueError(f"Unsupported heuristic level: {level}")
    if checkpoint_b is None and len(heuristic_levels) == 0:
        raise ValueError("Provide --checkpoint-b, --heuristic-levels, or both.")

    league_path = Path(args.league_path)
    league = load_league_state(path=league_path)
    updates: list[dict[str, object]] = []

    if checkpoint_b is not None:
        participant_b_id = args.participant_b_id or _default_participant_id("ckpt", checkpoint_b)
        participant_b_name = args.participant_b_name or checkpoint_b.stem
        series_summary = run_match_results_to_summary(
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
            games=max(1, int(args.games)),
            device=resolved_device,
            mcts_sims=int(args.mcts_sims),
            c_puct=float(args.c_puct),
            seed=int(args.seed),
        )
        league_summary = apply_series_to_league(
            league=league,
            participant_a_id=participant_a_id,
            participant_a_name=participant_a_name,
            participant_b_id=participant_b_id,
            participant_b_name=participant_b_name,
            series_summary=series_summary,
            participant_a_artifact_path=str(checkpoint_a),
            participant_b_artifact_path=str(checkpoint_b),
        )
        updates.append(
            {
                "opponent_id": participant_b_id,
                "opponent_name": participant_b_name,
                "series": series_summary,
                "league": league_summary,
            },
        )

    anchor_games = max(1, int(args.anchor_games or args.games))
    for offset, heuristic_level in enumerate(heuristic_levels, start=1):
        series_summary = run_checkpoint_vs_heuristic_to_summary(
            checkpoint=checkpoint_a,
            heuristic_level=heuristic_level,
            games=anchor_games,
            device=resolved_device,
            mcts_sims=int(args.mcts_sims),
            c_puct=float(args.c_puct),
            seed=int(args.seed) + (offset * 10_000),
        )
        league_summary = apply_series_to_league(
            league=league,
            participant_a_id=participant_a_id,
            participant_a_name=participant_a_name,
            participant_b_id=f"heu:{heuristic_level}",
            participant_b_name=heuristic_level,
            series_summary=series_summary,
            participant_a_artifact_path=str(checkpoint_a),
        )
        updates.append(
            {
                "opponent_id": f"heu:{heuristic_level}",
                "opponent_name": heuristic_level,
                "series": series_summary,
                "league": league_summary,
            },
        )

    save_league_state(path=league_path, league=league)
    champion_id = choose_champion_id(league)
    output = {
        "league_path": str(league_path),
        "participant_a_id": participant_a_id,
        "device": resolved_device,
        "champion_id": champion_id,
        "updates": updates,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print("League updated")
    print(f"  league_path: {league_path}")
    print(f"  champion_id: {champion_id}")
    for update in updates:
        series = update["series"]
        league_summary = update["league"]
        print(f"  vs {update['opponent_name']}:")
        print(f"    games: {int(series['games'])}")
        print(f"    participant_a_score: {float(series['checkpoint_a_score']):.3f}")
        print(f"    participant_a_rating: {float(league_summary['participant_a_rating']):.1f}")
        print(f"    opponent_rating: {float(league_summary['participant_b_rating']):.1f}")


if __name__ == "__main__":
    main()
