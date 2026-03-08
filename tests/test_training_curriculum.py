from __future__ import annotations

import unittest

import numpy as np

from agents.heuristic import HEURISTIC_LEVELS
from training.curriculum import get_curriculum_mix, sample_opponent_from_curriculum


class TestTrainingCurriculum(unittest.TestCase):
    def test_mix_probabilities_sum_to_one_per_group(self) -> None:
        for iteration in (1, 6, 13, 25):
            mix = get_curriculum_mix(iteration)
            self.assertAlmostEqual(mix["self"] + mix["heuristic"] + mix["random"], 1.0, places=6)
            self.assertAlmostEqual(
                mix["heu_easy"]
                + mix["heu_normal"]
                + mix["heu_hard"]
                + mix["heu_apex"]
                + mix["heu_gambit"]
                + mix["heu_sentinel"],
                1.0,
                places=6,
            )

    def test_curriculum_progressively_increases_self_play(self) -> None:
        early = get_curriculum_mix(1)["self"]
        mid = get_curriculum_mix(20)["self"]
        late = get_curriculum_mix(70)["self"]
        self.assertLess(early, mid)
        self.assertLess(mid, late)

    def test_early_curriculum_prioritizes_strong_heuristics(self) -> None:
        mix = get_curriculum_mix(1)
        weak_share = mix["heu_easy"] + mix["heu_normal"]
        strong_share = (
            mix["heu_hard"]
            + mix["heu_apex"]
            + mix["heu_gambit"]
            + mix["heu_sentinel"]
        )

        self.assertLessEqual(mix["heu_easy"], 0.05)
        self.assertLessEqual(weak_share, 0.25)
        self.assertGreaterEqual(strong_share, 0.75)
        self.assertLessEqual(mix["random"], 0.03)

    def test_sampling_outputs_known_labels(self) -> None:
        rng = np.random.default_rng(seed=7)
        for iteration in (1, 10, 30):
            opp, lvl = sample_opponent_from_curriculum(rng=rng, iteration=iteration)
            self.assertIn(opp, ("self", "heuristic", "random"))
            self.assertIn(lvl, HEURISTIC_LEVELS)


if __name__ == "__main__":
    unittest.main()
