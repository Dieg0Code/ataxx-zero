from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from game.constants import OBSERVATION_CHANNELS

_MODEL_KWARG_KEYS = {
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
}
_INPUT_PROJ_KEYS = ("input_proj.weight", "model.input_proj.weight")


def extract_model_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_hparams = payload.get("hparams")
    if not isinstance(raw_hparams, dict):
        raw_hparams = payload.get("hyper_parameters", {})
    if not isinstance(raw_hparams, dict):
        return {}
    return {key: raw_hparams[key] for key in _MODEL_KWARG_KEYS if key in raw_hparams}


def extract_checkpoint_state_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    state_dict_obj = payload.get("state_dict")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("Checkpoint dictionary must contain key 'state_dict'.")
    return state_dict_obj


def adapt_state_dict_observation_channels(
    state_dict: Mapping[str, Any],
    *,
    target_channels: int = OBSERVATION_CHANNELS,
) -> dict[str, Any]:
    adapted = dict(state_dict)
    for key in _INPUT_PROJ_KEYS:
        weight_obj = adapted.get(key)
        if not isinstance(weight_obj, torch.Tensor) or weight_obj.ndim != 2:
            continue
        source_channels = int(weight_obj.shape[1])
        if source_channels == target_channels:
            return adapted
        if source_channels > target_channels:
            raise ValueError(
                "Checkpoint expects more observation channels than the current model supports.",
            )
        # Older checkpoints were trained with fewer board features. Zero-pad any
        # newly added channels so they preserve prior behavior until retrained.
        pad = weight_obj.new_zeros((weight_obj.shape[0], target_channels - source_channels))
        adapted[key] = torch.cat((weight_obj, pad), dim=1)
        return adapted
    return adapted


def pad_observation_channels(
    observation: np.ndarray,
    *,
    target_channels: int = OBSERVATION_CHANNELS,
) -> np.ndarray:
    if observation.ndim not in (3, 4):
        raise ValueError("Observation array must have 3 or 4 dimensions.")
    channel_axis = 0 if observation.ndim == 3 else 1
    source_channels = int(observation.shape[channel_axis])
    if source_channels == target_channels:
        return observation
    if source_channels > target_channels:
        raise ValueError(
            "Observation array has more channels than the current model supports.",
        )

    pad_shape = list(observation.shape)
    pad_shape[channel_axis] = target_channels - source_channels
    pad_block = np.zeros(tuple(pad_shape), dtype=observation.dtype)
    return np.concatenate((observation, pad_block), axis=channel_axis)


__all__ = [
    "adapt_state_dict_observation_channels",
    "extract_checkpoint_state_dict",
    "extract_model_kwargs",
    "pad_observation_channels",
]
