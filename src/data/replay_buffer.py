from __future__ import annotations

import random
from collections import deque

import numpy as np
import numpy.typing as npt

Observation = npt.NDArray[np.float32]
PolicyTarget = npt.NDArray[np.float32]
TrainingExample = tuple[Observation, PolicyTarget, float]


def _pick_indices_with_min_repeats(
    *,
    population: list[int],
    sample_n: int,
    rng: np.random.Generator,
    repeat_counts: list[int],
) -> list[int]:
    if sample_n <= 0 or len(population) == 0:
        return []

    picked: list[int] = []
    remaining = sample_n
    while remaining > 0:
        min_repeat_count = min(repeat_counts[idx] for idx in population)
        candidates = [
            idx
            for idx in population
            if repeat_counts[idx] == min_repeat_count
        ]
        order = rng.permutation(len(candidates))
        take_n = min(remaining, len(candidates))
        chosen = [candidates[int(i)] for i in order[:take_n]]
        for idx in chosen:
            repeat_counts[idx] += 1
        picked.extend(chosen)
        remaining -= take_n
    return picked


def sample_recent_mix(
    examples: list[TrainingExample],
    *,
    recent_fraction: float,
    recent_window_fraction: float,
    seed: int | None = None,
    sample_size: int | None = None,
) -> list[TrainingExample]:
    """
    Build a training set biased toward recent samples while keeping global coverage.

    The sampler avoids replacement whenever the requested mix fits in the available
    pools. If the requested recent quota is larger than the recent window, repeats
    are spread as evenly as possible instead of hammering the same few examples.
    """
    if len(examples) == 0:
        return []

    total = len(examples)
    sample_n = total if sample_size is None else max(1, min(int(sample_size), total))

    recent_window_size = max(1, round(total * recent_window_fraction))
    recent_n = round(sample_n * recent_fraction)
    recent_n = min(sample_n, max(0, recent_n))
    global_n = sample_n - recent_n

    rng = np.random.default_rng(seed=seed)
    repeat_counts = [0] * total
    recent_population = list(range(total - recent_window_size, total))
    picked_indices = _pick_indices_with_min_repeats(
        population=recent_population,
        sample_n=recent_n,
        rng=rng,
        repeat_counts=repeat_counts,
    )
    picked_indices.extend(
        _pick_indices_with_min_repeats(
            population=list(range(total)),
            sample_n=global_n,
            rng=rng,
            repeat_counts=repeat_counts,
        ),
    )
    if len(picked_indices) > 1:
        order = rng.permutation(len(picked_indices))
        picked_indices = [picked_indices[int(i)] for i in order]
    return [examples[idx] for idx in picked_indices]


class ReplayBuffer:
    """FIFO replay buffer for self-play training examples."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.buffer: deque[TrainingExample] = deque(maxlen=capacity)

    def save_game(self, examples: list[TrainingExample]) -> None:
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def get_all(self) -> list[TrainingExample]:
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def clear(self) -> None:
        self.buffer.clear()

    def get_stats(self) -> dict[str, float | int | bool]:
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "is_full": False,
                "avg_value": 0.0,
                "win_rate_p1": 0.0,
                "loss_rate_p1": 0.0,
                "draw_rate": 0.0,
            }

        values = np.asarray([ex[2] for ex in self.buffer], dtype=np.float32)
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "is_full": self.is_full(),
            "avg_value": float(np.mean(values)),
            "win_rate_p1": float(np.mean(values == 1.0)),
            "loss_rate_p1": float(np.mean(values == -1.0)),
            "draw_rate": float(np.mean(values == 0.0)),
        }
