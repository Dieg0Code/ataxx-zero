from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RATING = 1200.0
ELO_K_FACTOR = 24.0
ELO_DELTA_CAP = 30.0
LEAGUE_VERSION = 1

LeagueState = dict[str, Any]
LeagueEntry = dict[str, str | float | int]
SeriesSummary = dict[str, float | int | str]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_empty_league_state() -> LeagueState:
    return {
        "version": LEAGUE_VERSION,
        "updated_at": utcnow_iso(),
        "champion_id": None,
        "entries": {},
        "recent_series": [],
    }


def load_league_state(*, path: Path) -> LeagueState:
    if not path.exists():
        return create_empty_league_state()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid league payload: expected dictionary.")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Invalid league payload: expected 'entries' dictionary.")
    if "recent_series" not in payload or not isinstance(payload["recent_series"], list):
        payload["recent_series"] = []
    return payload


def save_league_state(*, path: Path, league: LeagueState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    league["updated_at"] = utcnow_iso()
    path.write_text(json.dumps(league, indent=2, sort_keys=True), encoding="utf-8")


def compute_expected_score(*, rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _clamp_delta(delta: float) -> float:
    return max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, delta))


def _ensure_entry(
    *,
    league: LeagueState,
    participant_id: str,
    display_name: str,
    artifact_path: str | None = None,
) -> LeagueEntry:
    entries = league["entries"]
    existing = entries.get(participant_id)
    if isinstance(existing, dict):
        existing["display_name"] = display_name
        if artifact_path is not None and existing.get("kind") == "checkpoint":
            existing["artifact_path"] = artifact_path
        return existing

    kind = "heuristic" if participant_id.startswith("heu:") else "checkpoint"
    created: LeagueEntry = {
        "participant_id": participant_id,
        "display_name": display_name,
        "kind": kind,
        "rating": DEFAULT_RATING,
        "games_played": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }
    if kind == "checkpoint" and artifact_path is not None:
        created["artifact_path"] = artifact_path
    entries[participant_id] = created
    return created


def _apply_single_result(
    *,
    entry_a: LeagueEntry,
    entry_b: LeagueEntry,
    score_a: float,
) -> None:
    score_b = 1.0 - score_a
    rating_a = float(entry_a["rating"])
    rating_b = float(entry_b["rating"])
    expected_a = compute_expected_score(rating_a=rating_a, rating_b=rating_b)
    expected_b = 1.0 - expected_a
    delta_a = _clamp_delta(ELO_K_FACTOR * (score_a - expected_a))
    delta_b = _clamp_delta(ELO_K_FACTOR * (score_b - expected_b))

    entry_a["rating"] = rating_a + delta_a
    entry_b["rating"] = rating_b + delta_b
    entry_a["games_played"] = int(entry_a["games_played"]) + 1
    entry_b["games_played"] = int(entry_b["games_played"]) + 1

    if score_a == 1.0:
        entry_a["wins"] = int(entry_a["wins"]) + 1
        entry_b["losses"] = int(entry_b["losses"]) + 1
    elif score_a == 0.0:
        entry_a["losses"] = int(entry_a["losses"]) + 1
        entry_b["wins"] = int(entry_b["wins"]) + 1
    else:
        entry_a["draws"] = int(entry_a["draws"]) + 1
        entry_b["draws"] = int(entry_b["draws"]) + 1


def choose_champion_id(league: LeagueState) -> str | None:
    entries = league.get("entries", {})
    if not isinstance(entries, dict) or len(entries) == 0:
        return None

    checkpoint_entries = [entry for entry in entries.values() if entry.get("kind") == "checkpoint"]
    ranked_source = checkpoint_entries if len(checkpoint_entries) > 0 else list(entries.values())
    ranked = sorted(
        ranked_source,
        key=lambda item: (
            float(item["rating"]),
            int(item["games_played"]),
            str(item["participant_id"]),
        ),
        reverse=True,
    )
    top = ranked[0]
    return str(top["participant_id"])


def apply_series_to_league(
    *,
    league: LeagueState,
    participant_a_id: str,
    participant_a_name: str,
    participant_b_id: str,
    participant_b_name: str,
    series_summary: SeriesSummary,
    participant_a_artifact_path: str | None = None,
    participant_b_artifact_path: str | None = None,
) -> dict[str, float | int | str | None]:
    games = int(series_summary["games"])
    wins_a = int(series_summary["checkpoint_a_wins"])
    wins_b = int(series_summary["checkpoint_b_wins"])
    draws = int(series_summary["draws"])
    if wins_a + wins_b + draws != games:
        raise ValueError("Series summary counts do not match number of games.")

    entry_a = _ensure_entry(
        league=league,
        participant_id=participant_a_id,
        display_name=participant_a_name,
        artifact_path=participant_a_artifact_path,
    )
    entry_b = _ensure_entry(
        league=league,
        participant_id=participant_b_id,
        display_name=participant_b_name,
        artifact_path=participant_b_artifact_path,
    )
    rating_a_before = float(entry_a["rating"])
    rating_b_before = float(entry_b["rating"])

    # Apply per-game Elo updates so rating drift matches the backend-style formula.
    for _ in range(wins_a):
        _apply_single_result(entry_a=entry_a, entry_b=entry_b, score_a=1.0)
    for _ in range(draws):
        _apply_single_result(entry_a=entry_a, entry_b=entry_b, score_a=0.5)
    for _ in range(wins_b):
        _apply_single_result(entry_a=entry_a, entry_b=entry_b, score_a=0.0)

    league["champion_id"] = choose_champion_id(league)
    league["updated_at"] = utcnow_iso()
    recent_series = league.setdefault("recent_series", [])
    if not isinstance(recent_series, list):
        raise ValueError("Invalid league payload: 'recent_series' must be a list.")
    recent_series.append(
        {
            "participant_a_id": participant_a_id,
            "participant_b_id": participant_b_id,
            "games": games,
            "checkpoint_a_wins": wins_a,
            "checkpoint_b_wins": wins_b,
            "draws": draws,
            "checkpoint_a_score": float(series_summary["checkpoint_a_score"]),
            "avg_turns": float(series_summary["avg_turns"]),
            "rating_a_before": rating_a_before,
            "rating_a_after": float(entry_a["rating"]),
            "rating_b_before": rating_b_before,
            "rating_b_after": float(entry_b["rating"]),
            "played_at": utcnow_iso(),
        },
    )
    if len(recent_series) > 50:
        del recent_series[:-50]

    return {
        "games": games,
        "participant_a_rating": float(entry_a["rating"]),
        "participant_b_rating": float(entry_b["rating"]),
        "champion_id": league["champion_id"],
    }


__all__ = [
    "DEFAULT_RATING",
    "ELO_DELTA_CAP",
    "ELO_K_FACTOR",
    "apply_series_to_league",
    "choose_champion_id",
    "compute_expected_score",
    "create_empty_league_state",
    "load_league_state",
    "save_league_state",
]
