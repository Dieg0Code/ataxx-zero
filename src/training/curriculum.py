from __future__ import annotations

import numpy as np

from agents.heuristic import HEURISTIC_LEVELS

CurriculumMix = dict[str, float]


def get_curriculum_mix(iteration: int) -> CurriculumMix:
    """Phase-based opponent mix tuned to avoid early self-play collapse."""
    if iteration <= 8:
        return {
            "self": 0.05,
            "heuristic": 0.90,
            "random": 0.05,
            "heu_easy": 0.25,
            "heu_normal": 0.45,
            "heu_hard": 0.25,
            "heu_apex": 0.03,
            "heu_gambit": 0.01,
            "heu_sentinel": 0.01,
        }
    if iteration <= 20:
        return {
            "self": 0.20,
            "heuristic": 0.76,
            "random": 0.04,
            "heu_easy": 0.08,
            "heu_normal": 0.32,
            "heu_hard": 0.45,
            "heu_apex": 0.08,
            "heu_gambit": 0.03,
            "heu_sentinel": 0.04,
        }
    if iteration <= 45:
        return {
            "self": 0.40,
            "heuristic": 0.57,
            "random": 0.03,
            "heu_easy": 0.02,
            "heu_normal": 0.20,
            "heu_hard": 0.44,
            "heu_apex": 0.16,
            "heu_gambit": 0.08,
            "heu_sentinel": 0.10,
        }
    if iteration <= 90:
        return {
            "self": 0.55,
            "heuristic": 0.43,
            "random": 0.02,
            "heu_easy": 0.00,
            "heu_normal": 0.10,
            "heu_hard": 0.35,
            "heu_apex": 0.22,
            "heu_gambit": 0.13,
            "heu_sentinel": 0.20,
        }
    return {
        "self": 0.60,
        "heuristic": 0.38,
        "random": 0.02,
        "heu_easy": 0.00,
        "heu_normal": 0.08,
        "heu_hard": 0.30,
        "heu_apex": 0.24,
        "heu_gambit": 0.14,
        "heu_sentinel": 0.24,
    }


def sample_opponent_from_curriculum(
    rng: np.random.Generator,
    iteration: int,
) -> tuple[str, str]:
    mix = get_curriculum_mix(iteration)

    opp_labels = ("self", "heuristic", "random")
    opp_probs = np.asarray([mix["self"], mix["heuristic"], mix["random"]], dtype=np.float64)
    opp_probs = opp_probs / float(np.sum(opp_probs))
    opponent_type = str(rng.choice(opp_labels, p=opp_probs))

    heu_labels = HEURISTIC_LEVELS
    heu_probs = np.asarray([mix[f"heu_{level}"] for level in heu_labels], dtype=np.float64)
    heu_probs = heu_probs / float(np.sum(heu_probs))
    heuristic_level = str(rng.choice(heu_labels, p=heu_probs))
    return opponent_type, heuristic_level
