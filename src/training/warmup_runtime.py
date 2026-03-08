from __future__ import annotations

from agents.heuristic import is_supported_heuristic_level


def resolve_warmup_levels(*, raw_levels_csv: str, fallback_level: str) -> list[str]:
    """
    Parse warmup heuristic levels from CSV, dedupe preserving order, and validate.
    """
    levels: list[str] = []
    for token in raw_levels_csv.split(","):
        level = token.strip()
        if level == "" or level in levels:
            continue
        if not is_supported_heuristic_level(level):
            raise ValueError(f"Unsupported warmup heuristic level: {level}")
        levels.append(level)
    if len(levels) > 0:
        return levels
    if not is_supported_heuristic_level(fallback_level):
        raise ValueError(f"Unsupported warmup heuristic fallback level: {fallback_level}")
    return [fallback_level]


def allocate_warmup_games(*, total_games: int, levels: list[str]) -> list[tuple[str, int]]:
    """
    Evenly distribute warmup games across levels, assigning remainder to early levels.
    """
    if total_games <= 0 or len(levels) == 0:
        return []
    base = total_games // len(levels)
    remainder = total_games % len(levels)
    allocation: list[tuple[str, int]] = []
    for idx, level in enumerate(levels):
        games = base + (1 if idx < remainder else 0)
        allocation.append((level, games))
    return allocation


__all__ = [
    "allocate_warmup_games",
    "resolve_warmup_levels",
]
