from __future__ import annotations

import numpy as np

CurriculumMix = dict[str, float]


def get_curriculum_mix(iteration: int) -> CurriculumMix:
    """Phase-based opponent mix: start guided, then gradually increase self-play."""
    if iteration <= 5:
        return {
            "self": 0.00,
            "heuristic": 0.90,
            "random": 0.10,
            "heu_easy": 0.40,
            "heu_normal": 0.40,
            "heu_hard": 0.20,
        }
    if iteration <= 12:
        return {
            "self": 0.20,
            "heuristic": 0.70,
            "random": 0.10,
            "heu_easy": 0.10,
            "heu_normal": 0.40,
            "heu_hard": 0.50,
        }
    if iteration <= 20:
        return {
            "self": 0.45,
            "heuristic": 0.50,
            "random": 0.05,
            "heu_easy": 0.00,
            "heu_normal": 0.25,
            "heu_hard": 0.75,
        }
    return {
        "self": 0.70,
        "heuristic": 0.28,
        "random": 0.02,
        "heu_easy": 0.00,
        "heu_normal": 0.10,
        "heu_hard": 0.90,
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

    heu_labels = ("easy", "normal", "hard")
    heu_probs = np.asarray(
        [mix["heu_easy"], mix["heu_normal"], mix["heu_hard"]],
        dtype=np.float64,
    )
    heu_probs = heu_probs / float(np.sum(heu_probs))
    heuristic_level = str(rng.choice(heu_labels, p=heu_probs))
    return opponent_type, heuristic_level
