from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.constants import OBSERVATION_CHANNELS
from model.checkpoint_compat import (
    adapt_state_dict_observation_channels,
    pad_observation_channels,
)


class TestModelCheckpointCompat(unittest.TestCase):
    def test_adapt_state_dict_observation_channels_pads_new_channels(self) -> None:
        weight = torch.randn(32, 4)
        state_dict = {"model.input_proj.weight": weight.clone()}

        adapted = adapt_state_dict_observation_channels(state_dict)

        self.assertEqual(tuple(adapted["model.input_proj.weight"].shape), (32, OBSERVATION_CHANNELS))
        self.assertTrue(torch.allclose(adapted["model.input_proj.weight"][:, :4], weight))
        self.assertTrue(
            torch.allclose(
                adapted["model.input_proj.weight"][:, 4:],
                torch.zeros_like(adapted["model.input_proj.weight"][:, 4:]),
            )
        )

    def test_pad_observation_channels_extends_batch_arrays(self) -> None:
        observations = np.ones((3, 4, 7, 7), dtype=np.float32)

        padded = pad_observation_channels(observations)

        self.assertEqual(padded.shape, (3, OBSERVATION_CHANNELS, 7, 7))
        self.assertTrue(np.array_equal(padded[:, :4], observations))
        self.assertTrue(np.allclose(padded[:, 4:], 0.0))


if __name__ == "__main__":
    unittest.main()
