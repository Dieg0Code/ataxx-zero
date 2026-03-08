from __future__ import annotations

import numpy as np

from agents.heuristic import HEURISTIC_LEVELS

CurriculumMix = dict[str, float]


def get_curriculum_mix(iteration: int) -> CurriculumMix:
    """Phase-based opponent mix tuned to avoid early self-play collapse."""
    if iteration <= 12:
        return {
            "self": 0.10,
            "heuristic": 0.88,
            "random": 0.02,
            "heu_easy": 0.02,
            "heu_normal": 0.18,
            "heu_hard": 0.42,
            "heu_apex": 0.18,
            "heu_gambit": 0.08,
            "heu_sentinel": 0.12,
        }
    if iteration <= 30:
        return {
            "self": 0.25,
            "heuristic": 0.72,
            "random": 0.03,
            "heu_easy": 0.00,
            "heu_normal": 0.10,
            "heu_hard": 0.38,
            "heu_apex": 0.22,
            "heu_gambit": 0.12,
            "heu_sentinel": 0.18,
        }
    if iteration <= 60:
        return {
            "self": 0.45,
            "heuristic": 0.53,
            "random": 0.03,
            "heu_easy": 0.00,
            "heu_normal": 0.06,
            "heu_hard": 0.34,
            "heu_apex": 0.24,
            "heu_gambit": 0.14,
            "heu_sentinel": 0.22,
        }
    if iteration <= 90:
        return {
            "self": 0.58,
            "heuristic": 0.40,
            "random": 0.02,
            "heu_easy": 0.00,
            "heu_normal": 0.04,
            "heu_hard": 0.28,
            "heu_apex": 0.26,
            "heu_gambit": 0.16,
            "heu_sentinel": 0.26,
        }
    return {
        "self": 0.65,
        "heuristic": 0.33,
        "random": 0.02,
        "heu_easy": 0.00,
        "heu_normal": 0.02,
        "heu_hard": 0.24,
        "heu_apex": 0.28,
        "heu_gambit": 0.18,
        "heu_sentinel": 0.28,
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
