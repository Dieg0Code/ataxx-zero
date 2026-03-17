from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference.checkpoint_league_runtime import (
    apply_series_to_league,
    choose_champion_id,
    load_league_state,
    save_league_state,
)
from model.checkpoint_compat import adapt_state_dict_observation_channels
from training.config_runtime import cfg_bool, cfg_float, cfg_int, cfg_str

ALLOWED_MODEL_HPARAMS = {
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
}


@dataclass(frozen=True)
class CheckpointPoolEntry:
    participant_id: str
    display_name: str
    artifact_path: Path
    rating: float
    games_played: int


@dataclass(frozen=True)
class CheckpointModelSpec:
    entry: CheckpointPoolEntry
    state_dict: dict[str, torch.Tensor]
    model_cfg: dict[str, int | float]


def league_enabled() -> bool:
    return cfg_bool("league_enabled")


def resolve_league_path() -> Path:
    return Path(cfg_str("league_path"))


def _normalize_model_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict_obj = payload.get("state_dict")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("Checkpoint payload missing state_dict dictionary.")

    model_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict_obj.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value.detach().cpu()
    if len(model_state) == 0:
        raise ValueError("Checkpoint payload does not contain model.* weights.")
    adapted = adapt_state_dict_observation_channels(model_state)
    normalized: dict[str, torch.Tensor] = {}
    for key, value in adapted.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = value.detach().cpu()
    return normalized


def _extract_model_cfg(payload: dict[str, Any]) -> dict[str, int | float]:
    raw_hparams = payload.get("hparams")
    if not isinstance(raw_hparams, dict):
        raw_hparams = payload.get("hyper_parameters", {})
    if not isinstance(raw_hparams, dict):
        raw_hparams = {}

    extracted = {
        key: raw_hparams[key]
        for key in ALLOWED_MODEL_HPARAMS
        if key in raw_hparams
    }
    if len(extracted) == len(ALLOWED_MODEL_HPARAMS):
        return extracted

    return {
        "d_model": cfg_int("d_model"),
        "nhead": cfg_int("nhead"),
        "num_layers": cfg_int("num_layers"),
        "dim_feedforward": cfg_int("dim_feedforward"),
        "dropout": cfg_float("dropout"),
    }


def load_checkpoint_model_spec(entry: CheckpointPoolEntry) -> CheckpointModelSpec:
    payload = torch.load(str(entry.artifact_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint payload: expected dictionary.")
    return CheckpointModelSpec(
        entry=entry,
        state_dict=_normalize_model_state_dict(payload),
        model_cfg=_extract_model_cfg(payload),
    )


def _sorted_checkpoint_entries(entries: dict[str, Any]) -> list[CheckpointPoolEntry]:
    resolved: list[CheckpointPoolEntry] = []
    for raw_entry in entries.values():
        if not isinstance(raw_entry, dict):
            continue
        if str(raw_entry.get("kind")) != "checkpoint":
            continue
        artifact_path_raw = raw_entry.get("artifact_path")
        if not isinstance(artifact_path_raw, str) or artifact_path_raw.strip() == "":
            continue
        artifact_path = Path(artifact_path_raw)
        if not artifact_path.exists():
            continue
        resolved.append(
            CheckpointPoolEntry(
                participant_id=str(raw_entry["participant_id"]),
                display_name=str(raw_entry["display_name"]),
                artifact_path=artifact_path,
                rating=float(raw_entry["rating"]),
                games_played=int(raw_entry["games_played"]),
            ),
        )
    resolved.sort(
        key=lambda entry: (entry.rating, entry.games_played, entry.participant_id),
        reverse=True,
    )
    return resolved


def load_checkpoint_pool_entries(
    *,
    current_checkpoint_path: Path | None,
) -> list[CheckpointPoolEntry]:
    if not league_enabled():
        return []
    league_path = resolve_league_path()
    league = load_league_state(path=league_path)
    current_resolved = current_checkpoint_path.resolve() if current_checkpoint_path is not None else None
    min_games = max(0, cfg_int("league_checkpoint_pool_min_games"))
    top_k = max(0, cfg_int("league_checkpoint_pool_top_k"))

    entries = []
    for entry in _sorted_checkpoint_entries(league.get("entries", {})):
        if current_resolved is not None and entry.artifact_path.resolve() == current_resolved:
            continue
        if entry.games_played < min_games:
            continue
        entries.append(entry)
        if top_k > 0 and len(entries) >= top_k:
            break
    return entries


def resolve_champion_entry(
    *,
    current_checkpoint_path: Path | None,
) -> CheckpointPoolEntry | None:
    if not league_enabled():
        return None
    league = load_league_state(path=resolve_league_path())
    champion_id = choose_champion_id(league)
    if champion_id is None:
        return None
    current_resolved = current_checkpoint_path.resolve() if current_checkpoint_path is not None else None
    for entry in _sorted_checkpoint_entries(league.get("entries", {})):
        if entry.participant_id != champion_id:
            continue
        if current_resolved is not None and entry.artifact_path.resolve() == current_resolved:
            return None
        return entry
    return None


def select_checkpoint_pool_entry(
    *,
    entries: list[CheckpointPoolEntry],
    rng: np.random.Generator,
) -> CheckpointPoolEntry | None:
    if len(entries) == 0:
        return None
    if len(entries) == 1:
        return entries[0]
    weights = np.asarray(
        [len(entries) - idx for idx, _entry in enumerate(entries)],
        dtype=np.float64,
    )
    weights = weights / float(np.sum(weights))
    picked_idx = int(rng.choice(len(entries), p=weights))
    return entries[picked_idx]


def build_checkpoint_pool_specs(
    *,
    current_checkpoint_path: Path | None,
) -> dict[str, CheckpointModelSpec]:
    specs: dict[str, CheckpointModelSpec] = {}
    for entry in load_checkpoint_pool_entries(current_checkpoint_path=current_checkpoint_path):
        specs[entry.participant_id] = load_checkpoint_model_spec(entry)
    return specs


def record_checkpoint_in_league(
    *,
    checkpoint_path: Path,
    heuristic_series_by_level: dict[str, dict[str, float | int | str]],
    champion_entry: CheckpointPoolEntry | None = None,
    champion_series_summary: dict[str, float | int | str] | None = None,
) -> dict[str, Any]:
    league_path = resolve_league_path()
    league = load_league_state(path=league_path)
    participant_a_id = f"ckpt:{checkpoint_path.stem}"
    participant_a_name = checkpoint_path.stem

    for level, series_summary in heuristic_series_by_level.items():
        apply_series_to_league(
            league=league,
            participant_a_id=participant_a_id,
            participant_a_name=participant_a_name,
            participant_b_id=f"heu:{level}",
            participant_b_name=level,
            series_summary=series_summary,
            participant_a_artifact_path=str(checkpoint_path),
        )

    if champion_entry is not None and champion_series_summary is not None:
        apply_series_to_league(
            league=league,
            participant_a_id=participant_a_id,
            participant_a_name=participant_a_name,
            participant_b_id=champion_entry.participant_id,
            participant_b_name=champion_entry.display_name,
            series_summary=champion_series_summary,
            participant_a_artifact_path=str(checkpoint_path),
            participant_b_artifact_path=str(champion_entry.artifact_path),
        )

    save_league_state(path=league_path, league=league)
    return league


__all__ = [
    "CheckpointModelSpec",
    "CheckpointPoolEntry",
    "build_checkpoint_pool_specs",
    "league_enabled",
    "load_checkpoint_model_spec",
    "load_checkpoint_pool_entries",
    "record_checkpoint_in_league",
    "resolve_champion_entry",
    "resolve_league_path",
    "select_checkpoint_pool_entry",
]
