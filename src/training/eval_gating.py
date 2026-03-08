from __future__ import annotations


def compute_regression_gate(
    *,
    current_score: float,
    best_score: float,
    regression_delta: float,
    current_streak: int,
    patience: int,
) -> tuple[int, bool]:
    """
    Update regression streak and indicate whether best-checkpoint restore is recommended.
    """
    if patience <= 0:
        return 0, False

    if current_score + max(0.0, regression_delta) < best_score:
        next_streak = current_streak + 1
    else:
        next_streak = 0
    should_restore = next_streak >= patience
    return next_streak, should_restore


__all__ = ["compute_regression_gate"]
