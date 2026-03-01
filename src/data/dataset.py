from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

from data.replay_buffer import ReplayBuffer
from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE
from game.types import Move

_N_TRANSFORMS = 8
_POLICY_INDEX_MAPS: np.ndarray | None = None


def _rotate_coord_ccw(r: int, c: int, k: int, size: int) -> tuple[int, int]:
    rr, cc = r, c
    for _ in range(k):
        rr, cc = size - 1 - cc, rr
    return rr, cc


def _flip_coord_horizontal(r: int, c: int, size: int) -> tuple[int, int]:
    return r, size - 1 - c


def _transform_move(move: Move | None, transform_id: int, size: int) -> Move | None:
    if move is None:
        return None

    r1, c1, r2, c2 = move
    if transform_id == 0:
        return move
    if 1 <= transform_id <= 3:
        k = transform_id
        nr1, nc1 = _rotate_coord_ccw(r1, c1, k, size)
        nr2, nc2 = _rotate_coord_ccw(r2, c2, k, size)
        return (nr1, nc1, nr2, nc2)

    fr1, fc1 = _flip_coord_horizontal(r1, c1, size)
    fr2, fc2 = _flip_coord_horizontal(r2, c2, size)
    if transform_id == 4:
        return (fr1, fc1, fr2, fc2)

    k = transform_id - 4
    nr1, nc1 = _rotate_coord_ccw(fr1, fc1, k, size)
    nr2, nc2 = _rotate_coord_ccw(fr2, fc2, k, size)
    return (nr1, nc1, nr2, nc2)


def _augment_observation(observation: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return observation
    if 1 <= transform_id <= 3:
        return np.rot90(observation, k=transform_id, axes=(1, 2)).copy()

    obs_aug = np.flip(observation, axis=2).copy()
    k = transform_id - 4
    if k > 0:
        obs_aug = np.rot90(obs_aug, k=k, axes=(1, 2)).copy()
    return obs_aug


def _get_policy_index_maps() -> np.ndarray:
    """Lazily build action-index maps for each board symmetry transform."""
    global _POLICY_INDEX_MAPS
    if _POLICY_INDEX_MAPS is not None:
        return _POLICY_INDEX_MAPS

    maps = np.zeros((_N_TRANSFORMS, ACTION_SPACE.num_actions), dtype=np.int64)
    for transform_id in range(_N_TRANSFORMS):
        for action_idx in range(ACTION_SPACE.num_actions):
            move = ACTION_SPACE.decode(action_idx)
            transformed_move = _transform_move(
                move=move,
                transform_id=transform_id,
                size=BOARD_SIZE,
            )
            maps[transform_id, action_idx] = ACTION_SPACE.encode(transformed_move)

    _POLICY_INDEX_MAPS = maps
    return _POLICY_INDEX_MAPS


def _augment_policy(policy: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return policy

    index_map = _get_policy_index_maps()[transform_id]
    pi_aug = np.zeros_like(policy)
    np.add.at(pi_aug, index_map, policy)

    total = float(np.sum(pi_aug))
    if total > 0.0:
        pi_aug /= total
    return pi_aug


class AtaxxDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset wrapper from replay buffer examples."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        augment: bool = True,
        reference_buffer: bool = False,
        val_split: float = 0.1,
    ) -> None:
        self.augment = augment
        self.examples: list[tuple[np.ndarray, np.ndarray, float]] | deque[
            tuple[np.ndarray, np.ndarray, float]
        ]
        raw_examples = list(buffer.buffer) if reference_buffer else buffer.get_all()
        n_val = int(len(raw_examples) * val_split)
        n_train = len(raw_examples) - n_val
        # Keep train/validation disjoint so val loss is a true hold-out metric.
        self.examples = raw_examples[:n_train]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation, policy, value = self.examples[index]

        transform_id = 0
        if self.augment:
            transform_id = int(np.random.randint(0, _N_TRANSFORMS))
        if transform_id != 0:
            observation = _augment_observation(observation, transform_id)
            policy = _augment_policy(policy, transform_id)

        board_tensor = torch.from_numpy(observation).float()
        pi_tensor = torch.from_numpy(policy).float()
        value_tensor = torch.tensor(value, dtype=torch.float32)
        return board_tensor, pi_tensor, value_tensor


class ValidationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Hold-out validation split from replay buffer."""

    def __init__(self, buffer: ReplayBuffer, split: float = 0.1) -> None:
        all_examples = buffer.get_all()
        n_val = int(len(all_examples) * split)
        n_train = len(all_examples) - n_val
        self.examples = all_examples[n_train:] if n_val > 0 else []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation, policy, value = self.examples[index]
        board_tensor = torch.from_numpy(observation).float()
        pi_tensor = torch.from_numpy(policy).float()
        value_tensor = torch.tensor(value, dtype=torch.float32)
        return board_tensor, pi_tensor, value_tensor
