from __future__ import annotations

import random
from collections import deque

import numpy as np
import numpy.typing as npt

Observation = npt.NDArray[np.float32]
PolicyTarget = npt.NDArray[np.float32]
TrainingExample = tuple[Observation, PolicyTarget, float]


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

    The default behavior keeps dataset size unchanged and mixes:
    - `recent_fraction` from the most recent `recent_window_fraction` of samples,
    - the rest from the full training pool.
    """
    if len(examples) == 0:
        return []

    total = len(examples)
    sample_n = total if sample_size is None else max(1, min(int(sample_size), total))

    recent_window_size = max(1, round(total * recent_window_fraction))
    recent_window = examples[-recent_window_size:]
    recent_n = round(sample_n * recent_fraction)
    recent_n = min(sample_n, max(0, recent_n))
    global_n = sample_n - recent_n

    rng = np.random.default_rng(seed=seed)
    picked: list[TrainingExample] = []
    if recent_n > 0:
        recent_idx = rng.integers(0, len(recent_window), size=recent_n, endpoint=False)
        picked.extend(recent_window[int(i)] for i in recent_idx)
    if global_n > 0:
        global_idx = rng.integers(0, total, size=global_n, endpoint=False)
        picked.extend(examples[int(i)] for i in global_idx)
    if len(picked) > 1:
        order = rng.permutation(len(picked))
        picked = [picked[int(i)] for i in order]
    return picked


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
