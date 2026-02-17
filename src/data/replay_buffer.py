from __future__ import annotations

import random
from collections import deque

import numpy as np
import numpy.typing as npt

Observation = npt.NDArray[np.float32]
PolicyTarget = npt.NDArray[np.float32]
TrainingExample = tuple[Observation, PolicyTarget, float]


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
