from __future__ import annotations

import numpy as np

from game.board import AtaxxBoard
from game.types import Move


def _score_move(state: AtaxxBoard, move: Move) -> float:
    r1, c1, r2, c2 = move
    me = state.current_player
    before_me = int(np.sum(state.grid == me))
    before_opp = int(np.sum(state.grid == -me))
    scratch = state.copy()
    scratch.step(move)
    after_me = int(np.sum(scratch.grid == me))
    after_opp = int(np.sum(scratch.grid == -me))
    clone_bonus = 0.15 if max(abs(r1 - r2), abs(c1 - c2)) == 1 else 0.0
    center_bonus = 0.05 * (3 - abs(r2 - 3) + 3 - abs(c2 - 3))
    return float((after_me - before_me) + (before_opp - after_opp)) + clone_bonus + center_bonus


def heuristic_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
    level: str = "normal",
) -> Move | None:
    valid_moves = board.get_valid_moves()
    if len(valid_moves) == 0:
        return None

    if level == "easy":
        scores = np.asarray([_score_move(board, move) for move in valid_moves], dtype=np.float32)
        scores = scores - float(np.min(scores)) + 0.2
        probs = scores / float(np.sum(scores))
        return valid_moves[int(rng.choice(len(valid_moves), p=probs))]

    scored_moves: list[tuple[Move, float]] = []
    for move in valid_moves:
        score = _score_move(board, move)
        if level == "hard":
            scratch = board.copy()
            scratch.step(move)
            opp_moves = scratch.get_valid_moves()
            if len(opp_moves) > 0:
                opp_best = max(_score_move(scratch, opp_move) for opp_move in opp_moves)
                score -= 0.65 * opp_best
        scored_moves.append((move, score))

    if level == "normal":
        # Normal is deliberately non-greedy to avoid repetitive games.
        scores = np.asarray([score for _, score in scored_moves], dtype=np.float32)
        temperature = 0.35
        logits = (scores - float(np.max(scores))) / temperature
        probs = np.exp(logits)
        probs = probs / float(np.sum(probs))
        pick_idx = int(rng.choice(len(scored_moves), p=probs))
        return scored_moves[pick_idx][0]

    best_score = max(score for _, score in scored_moves)
    best_moves = [move for move, score in scored_moves if score == best_score]
    return best_moves[int(rng.integers(0, len(best_moves)))]
