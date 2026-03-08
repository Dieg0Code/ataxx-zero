from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from data.replay_buffer import ReplayBuffer
from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE
from game.types import Move

_N_TRANSFORMS = 8
_POLICY_INDEX_MAPS: np.ndarray | None = None

if TYPE_CHECKING:
    from data.replay_buffer import TrainingExample


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


def split_train_val_examples(
    *,
    all_examples: list[TrainingExample],
    val_split: float,
    shuffle: bool,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split examples into disjoint train/val sets with optional seeded shuffling."""
    n_total = len(all_examples)
    if n_total == 0:
        return [], []
    n_val = int(n_total * val_split)
    n_val = min(max(0, n_val), n_total)
    n_train = n_total - n_val
    if n_val == 0:
        return list(all_examples), []
    if not shuffle:
        return list(all_examples[:n_train]), list(all_examples[n_train:])

    rng = np.random.default_rng(seed=seed)
    val_indices = np.sort(rng.choice(n_total, size=n_val, replace=False))
    val_set = {int(i) for i in val_indices.tolist()}
    # Keep train set in chronological order so "recent" remains meaningful.
    train_indices = [idx for idx in range(n_total) if idx not in val_set]
    train_examples = [all_examples[idx] for idx in train_indices]
    val_examples = [all_examples[int(idx)] for idx in val_indices]
    return train_examples, val_examples


class AtaxxDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset wrapper from replay buffer examples."""

    def __init__(
        self,
        buffer: ReplayBuffer | None = None,
        augment: bool = True,
        reference_buffer: bool = False,
        val_split: float = 0.1,
        examples: list[TrainingExample] | None = None,
    ) -> None:
        self.augment = augment
        self.examples: list[tuple[np.ndarray, np.ndarray, float]] | deque[
            tuple[np.ndarray, np.ndarray, float]
        ]
        if examples is not None:
            self.examples = list(examples)
            return
        if buffer is None:
            self.examples = []
            return

        raw_examples = list(buffer.buffer) if reference_buffer else buffer.get_all()
        train_examples, _ = split_train_val_examples(
            all_examples=raw_examples,
            val_split=val_split,
            shuffle=False,
            seed=0,
        )
        self.examples = train_examples

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

    def __init__(
        self,
        buffer: ReplayBuffer | None = None,
        split: float = 0.1,
        examples: list[TrainingExample] | None = None,
    ) -> None:
        if examples is not None:
            self.examples = list(examples)
            return
        if buffer is None:
            self.examples = []
            return
        all_examples = buffer.get_all()
        _, val_examples = split_train_val_examples(
            all_examples=all_examples,
            val_split=split,
            shuffle=False,
            seed=0,
        )
        self.examples = val_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation, policy, value = self.examples[index]
        board_tensor = torch.from_numpy(observation).float()
        pi_tensor = torch.from_numpy(policy).float()
        value_tensor = torch.tensor(value, dtype=torch.float32)
        return board_tensor, pi_tensor, value_tensor
