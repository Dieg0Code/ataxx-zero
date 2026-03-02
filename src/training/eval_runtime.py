from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from game.actions import ACTION_SPACE
from training.config_runtime import cfg_bool, cfg_int
from training.selfplay_runtime import compute_action_probs, heuristic_move

if TYPE_CHECKING:
    from engine.mcts import MCTS
    from model.system import AtaxxZero


def _play_eval_episode(
    mcts: MCTS,
    rng: np.random.Generator,
    heuristic_level: str,
) -> int:
    from game.board import AtaxxBoard

    board = AtaxxBoard()
    root = None
    model_player = 1 if float(rng.random()) >= 0.5 else -1
    while not board.is_game_over():
        if board.current_player == model_player:
            probs, root = compute_action_probs(
                board=board,
                mcts=mcts,
                root=root,
                add_noise=False,
                temperature=0.0,
            )
            action_idx = int(np.argmax(probs))
            board.step(ACTION_SPACE.decode(action_idx))
            root = mcts.advance_root(root, action_idx)
            continue
        move = heuristic_move(board, rng, heuristic_level)
        board.step(move)
        root = mcts.advance_root(root, ACTION_SPACE.encode(move))
    winner = board.get_result()
    if winner == model_player:
        return 1
    if winner == 0:
        return 0
    return -1


def evaluate_model(
    system: AtaxxZero,
    device: str,
    games: int,
    sims: int,
    c_puct: float,
    heuristic_level: str,
    seed: int,
) -> dict[str, float | int | str]:
    from engine.mcts import MCTS

    system.eval()
    system.to(device)
    mcts = MCTS(
        model=system.model,
        c_puct=c_puct,
        n_simulations=sims,
        device=device,
        use_amp=cfg_bool("mcts_use_amp"),
        cache_size=max(0, cfg_int("mcts_cache_size")),
        leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
    )
    rng = np.random.default_rng(seed=seed)
    wins = 0
    losses = 0
    draws = 0
    for _ in range(games):
        outcome = _play_eval_episode(mcts, rng, heuristic_level)
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1
    score = (wins + 0.5 * draws) / max(1, games)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "heuristic_level": heuristic_level,
        "sims": sims,
    }


__all__ = [
    "evaluate_model",
]
