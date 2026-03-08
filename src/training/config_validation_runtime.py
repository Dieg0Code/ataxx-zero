from __future__ import annotations

from collections.abc import Callable

from agents.heuristic import is_supported_heuristic_level


def validate_reward_shaping_config(cfg_float: Callable[[str], float]) -> None:
    if cfg_float("reward_shaping_scale") < 0.0:
        raise ValueError("CONFIG['reward_shaping_scale'] must be >= 0.")
    reward_gamma = cfg_float("reward_shaping_gamma")
    if not 0.0 <= reward_gamma <= 1.0:
        raise ValueError("CONFIG['reward_shaping_gamma'] must be in [0, 1].")
    if cfg_float("reward_shaping_material_weight") < 0.0:
        raise ValueError("CONFIG['reward_shaping_material_weight'] must be >= 0.")
    if cfg_float("reward_shaping_mobility_weight") < 0.0:
        raise ValueError("CONFIG['reward_shaping_mobility_weight'] must be >= 0.")
    if cfg_float("reward_shaping_draw_penalty") < 0.0:
        raise ValueError("CONFIG['reward_shaping_draw_penalty'] must be >= 0.")


def validate_supported_heuristic_csv(*, raw_levels: str, setting_name: str) -> None:
    if raw_levels == "":
        return
    for level in [part.strip() for part in raw_levels.split(",") if part.strip()]:
        if not is_supported_heuristic_level(level):
            raise ValueError(
                f"CONFIG['{setting_name}'] contains unsupported level '{level}'.",
            )


__all__ = [
    "validate_reward_shaping_config",
    "validate_supported_heuristic_csv",
]
