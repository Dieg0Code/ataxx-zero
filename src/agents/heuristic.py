from __future__ import annotations

import numpy as np

from game.board import AtaxxBoard
from game.types import Move

HEURISTIC_LEVELS: tuple[str, ...] = (
    "easy",
    "normal",
    "hard",
    "apex",
    "gambit",
    "sentinel",
)
HEURISTIC_LEVEL_SET = frozenset(HEURISTIC_LEVELS)
DEFAULT_HEURISTIC_LEVEL = "normal"


def is_supported_heuristic_level(level: str) -> bool:
    return level in HEURISTIC_LEVEL_SET


def heuristic_mode_from_level(level: str) -> str:
    if not is_supported_heuristic_level(level):
        raise ValueError(f"Unsupported heuristic level: {level}")
    return f"heuristic_{level}"


def _chebyshev_distance(move: Move) -> int:
    r1, c1, r2, c2 = move
    return max(abs(r1 - r2), abs(c1 - c2))


def _count_targets_in_radius(
    board: AtaxxBoard,
    *,
    row: int,
    col: int,
    target: int,
    radius: int,
) -> int:
    board_size = board.grid.shape[0]
    r_min = max(0, row - radius)
    r_max = min(board_size, row + radius + 1)
    c_min = max(0, col - radius)
    c_max = min(board_size, col + radius + 1)
    window = board.grid[r_min:r_max, c_min:c_max]
    return int(np.sum(window == target))


def _mobility_advantage(after_move: AtaxxBoard) -> float:
    opponent_moves = len(after_move.get_valid_moves(player=after_move.current_player))
    own_moves = len(after_move.get_valid_moves(player=-after_move.current_player))
    return float(own_moves - opponent_moves)


def _score_move(state: AtaxxBoard, move: Move) -> float:
    _, _, r2, c2 = move
    me = state.current_player
    before_me = int(np.sum(state.grid == me))
    before_opp = int(np.sum(state.grid == -me))
    scratch = state.copy()
    scratch.step(move)
    after_me = int(np.sum(scratch.grid == me))
    after_opp = int(np.sum(scratch.grid == -me))
    clone_bonus = 0.15 if _chebyshev_distance(move) == 1 else 0.0
    center_bonus = 0.05 * (3 - abs(r2 - 3) + 3 - abs(c2 - 3))
    return float((after_me - before_me) + (before_opp - after_opp)) + clone_bonus + center_bonus


def _best_reply_penalty(after_move: AtaxxBoard) -> float:
    opp_moves = after_move.get_valid_moves()
    if len(opp_moves) == 0:
        return -2.0
    return float(max(_score_move(after_move, opp_move) for opp_move in opp_moves))


def _softmax_choice(
    rng: np.random.Generator,
    scored_moves: list[tuple[Move, float]],
    *,
    temperature: float,
) -> Move:
    scores = np.asarray([score for _, score in scored_moves], dtype=np.float32)
    logits = (scores - float(np.max(scores))) / temperature
    probs = np.exp(logits)
    probs = probs / float(np.sum(probs))
    pick_idx = int(rng.choice(len(scored_moves), p=probs))
    return scored_moves[pick_idx][0]


def _score_apex(board: AtaxxBoard, move: Move) -> float:
    base = _score_move(board, move)
    after = board.copy()
    after.step(move)
    opp_moves = after.get_valid_moves()
    mobility = _mobility_advantage(after)
    if len(opp_moves) == 0:
        return base + 3.0 + 0.2 * mobility

    # Two-ply selective lookahead: punish lines where opponent can spike value
    # and we fail to recover with a strong counter on the next turn.
    opp_candidates = sorted(
        opp_moves,
        key=lambda opp_move: _score_move(after, opp_move),
        reverse=True,
    )[:3]
    worst_line = float("-inf")
    for opp_move in opp_candidates:
        reply_board = after.copy()
        reply_board.step(opp_move)
        reply_moves = reply_board.get_valid_moves()
        reply_best = (
            max(_score_move(reply_board, reply_move) for reply_move in reply_moves)
            if len(reply_moves) > 0
            else -2.5
        )
        line_value = _score_move(after, opp_move) - 0.55 * float(reply_best)
        worst_line = max(worst_line, float(line_value))

    return base - 0.92 * worst_line + 0.2 * mobility


def _score_gambit(board: AtaxxBoard, move: Move) -> float:
    _, _, r2, c2 = move
    base = _score_move(board, move)
    after = board.copy()
    after.step(move)
    enemy = after.current_player
    frontier_risk = _count_targets_in_radius(
        after,
        row=r2,
        col=c2,
        target=enemy,
        radius=1,
    )
    pressure_ring = _count_targets_in_radius(
        after,
        row=r2,
        col=c2,
        target=enemy,
        radius=2,
    )
    jump_bonus = 0.55 if _chebyshev_distance(move) == 2 else -0.12
    flank_bonus = 0.35 if r2 in {0, 6} or c2 in {0, 6} else 0.0
    hard_guard = _best_reply_penalty(after)
    return (
        base
        - 0.58 * hard_guard
        + 0.46 * float(pressure_ring)
        + jump_bonus
        + flank_bonus
        - 0.42 * float(frontier_risk)
    )


def _score_sentinel(board: AtaxxBoard, move: Move) -> float:
    _, _, r2, c2 = move
    base = _score_move(board, move)
    after = board.copy()
    after.step(move)
    enemy = after.current_player
    own_piece = -enemy
    frontier_risk = _count_targets_in_radius(
        after,
        row=r2,
        col=c2,
        target=enemy,
        radius=1,
    )
    local_support = (
        _count_targets_in_radius(
            after,
            row=r2,
            col=c2,
            target=own_piece,
            radius=1,
        )
        - 1
    )
    mobility = _mobility_advantage(after)
    center_bonus = 0.18 * (3 - abs(r2 - 3) + 3 - abs(c2 - 3))
    clone_bias = 0.4 if _chebyshev_distance(move) == 1 else -0.06
    hard_guard = _best_reply_penalty(after)
    return (
        base
        - 0.56 * hard_guard
        + 0.34 * mobility
        + 0.36 * float(local_support)
        + center_bonus
        + clone_bias
        - 0.5 * float(frontier_risk)
    )


def heuristic_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
    level: str = DEFAULT_HEURISTIC_LEVEL,
) -> Move | None:
    if not is_supported_heuristic_level(level):
        raise ValueError(f"Unsupported heuristic level: {level}")

    valid_moves = board.get_valid_moves()
    if len(valid_moves) == 0:
        return None

    if level == "easy":
        scored_moves = [(move, _score_move(board, move)) for move in valid_moves]
        # Easy should still punish obvious blunders while keeping variety.
        return _softmax_choice(rng, scored_moves, temperature=0.85)

    scored_moves: list[tuple[Move, float]] = []
    for move in valid_moves:
        score = _score_move(board, move)
        if level == "hard":
            scratch = board.copy()
            scratch.step(move)
            score -= 0.65 * _best_reply_penalty(scratch)
            score += 0.12 * _mobility_advantage(scratch)
        elif level == "apex":
            score = _score_apex(board, move)
        elif level == "gambit":
            score = _score_gambit(board, move)
        elif level == "sentinel":
            score = _score_sentinel(board, move)
        scored_moves.append((move, score))

    if level == "normal":
        # Normal is deliberately non-greedy to avoid repetitive games.
        return _softmax_choice(rng, scored_moves, temperature=0.35)

    best_score = max(score for _, score in scored_moves)
    best_moves = [move for move, score in scored_moves if score == best_score]
    return best_moves[int(rng.integers(0, len(best_moves)))]
