from __future__ import annotations

from collections.abc import Callable

DEFAULT_LEAGUE_CONFIG: dict[str, int | float | bool | str] = {
    "league_enabled": True,
    "league_path": "checkpoints/league_ratings.json",
    "league_checkpoint_pool_top_k": 3,
    "league_checkpoint_pool_min_games": 4,
    "league_selfplay_checkpoint_prob": 0.35,
    "league_champion_games": 6,
}


def validate_league_config(
    *,
    cfg_int: Callable[[str], int],
    cfg_float: Callable[[str], float],
) -> None:
    if cfg_int("league_checkpoint_pool_top_k") < 0:
        raise ValueError("CONFIG['league_checkpoint_pool_top_k'] must be >= 0.")
    if cfg_int("league_checkpoint_pool_min_games") < 0:
        raise ValueError("CONFIG['league_checkpoint_pool_min_games'] must be >= 0.")
    if not 0.0 <= cfg_float("league_selfplay_checkpoint_prob") <= 1.0:
        raise ValueError("CONFIG['league_selfplay_checkpoint_prob'] must be in [0, 1].")
    if cfg_int("league_champion_games") < 0:
        raise ValueError("CONFIG['league_champion_games'] must be >= 0.")
