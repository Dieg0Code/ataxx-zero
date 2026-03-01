from __future__ import annotations

import heapq
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from game.actions import ACTION_SPACE
from training.config_runtime import (
    cfg_bool,
    cfg_float,
    cfg_int,
    log,
)

if TYPE_CHECKING:
    from data.replay_buffer import ReplayBuffer, TrainingExample
    from engine.mcts import MCTS, MCTSNode
    from game.board import AtaxxBoard
    from model.system import AtaxxZero

_WORKER_MCTS: object | None = None


def compute_action_probs(
    board: AtaxxBoard,
    mcts: MCTS,
    root: MCTSNode | None,
    add_noise: bool,
    temperature: float,
) -> tuple[np.ndarray, MCTSNode | None]:
    probs, updated_root = mcts.run_with_root(
        board=board,
        root=root,
        add_dirichlet_noise=add_noise,
        temperature=temperature,
    )
    total_prob = float(np.sum(probs))
    if total_prob > 0.0:
        return probs, updated_root

    valid_moves = board.get_valid_moves()
    fallback = ACTION_SPACE.mask_from_moves(
        valid_moves,
        include_pass=(len(valid_moves) == 0),
    )
    return fallback / float(np.sum(fallback)), updated_root


def select_action_idx(
    probs: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature > 0.0:
        return int(rng.choice(len(probs), p=probs))
    return int(np.argmax(probs))


def score_move_for_player(board: AtaxxBoard, move: tuple[int, int, int, int]) -> float:
    r1, c1, r2, c2 = move
    board_size = int(board.grid.shape[0])
    jump = 1 if max(abs(r2 - r1), abs(c2 - c1)) == 2 else 0
    r_min = max(0, r2 - 1)
    r_max = min(board_size, r2 + 2)
    c_min = max(0, c2 - 1)
    c_max = min(board_size, c2 + 2)
    neighborhood = board.grid[r_min:r_max, c_min:c_max]
    converted = float(np.sum(neighborhood == -board.current_player))
    center = float(board_size - 1) / 2.0
    center_bonus = 0.35 * ((board_size - 1) - abs(r2 - center) - abs(c2 - center))
    return converted + center_bonus - 0.55 * float(jump)


def heuristic_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
    level: str,
) -> tuple[int, int, int, int] | None:
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return None
    if level == "easy":
        return moves[int(rng.integers(0, len(moves)))]

    if level == "hard":
        top_k = max(1, min(3, len(moves)))
    else:
        top_k = max(1, min(5, len(moves)))
    scored = heapq.nlargest(
        top_k,
        ((move, score_move_for_player(board, move)) for move in moves),
        key=lambda item: item[1],
    )
    weights = np.linspace(1.0, 0.35, top_k, dtype=np.float64)
    weights = weights / np.sum(weights)
    pick = int(rng.choice(top_k, p=weights))
    return scored[pick][0]


def random_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return None
    return moves[int(rng.integers(0, len(moves)))]


def play_episode(
    mcts: MCTS,
    add_noise: bool,
    temp_threshold: int,
    rng: np.random.Generator,
    opponent_type: str,
    opponent_heuristic_level: str,
    model_player: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int, int]:
    from game.board import AtaxxBoard

    board = AtaxxBoard()
    root = None
    game_history: list[tuple[np.ndarray, np.ndarray, int]] = []
    turn_idx = 0

    while not board.is_game_over():
        turn_idx += 1
        is_model_turn = board.current_player == model_player
        if is_model_turn or opponent_type == "self":
            temperature = 1.0 if turn_idx <= temp_threshold else 0.0
            use_noise = add_noise and (is_model_turn or opponent_type == "self")
            probs, root = compute_action_probs(
                board=board,
                mcts=mcts,
                root=root,
                add_noise=use_noise,
                temperature=temperature,
            )
            game_history.append(
                (
                    board.get_observation(),
                    probs,
                    board.current_player,
                )
            )
            action_idx = select_action_idx(
                probs=probs,
                temperature=temperature,
                rng=rng,
            )
            board.step(ACTION_SPACE.decode(action_idx))
            root = mcts.advance_root(root, action_idx)
            continue

        if opponent_type == "heuristic":
            move = heuristic_move(board, rng, opponent_heuristic_level)
            board.step(move)
            root = mcts.advance_root(root, ACTION_SPACE.encode(move))
            continue

        move = random_move(board, rng)
        board.step(move)
        root = mcts.advance_root(root, ACTION_SPACE.encode(move))

    return game_history, board.get_result(), turn_idx


def _init_selfplay_process_worker(
    model_state_dict: dict[str, torch.Tensor],
    model_cfg: dict[str, int | float],
    c_puct: float,
    sims: int,
) -> None:
    global _WORKER_MCTS
    from engine.mcts import MCTS
    from model.transformer import AtaxxTransformerNet

    model = AtaxxTransformerNet(
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    _WORKER_MCTS = MCTS(
        model=model,
        c_puct=c_puct,
        n_simulations=sims,
        device="cpu",
        use_amp=False,
        cache_size=max(0, cfg_int("mcts_cache_size")),
        leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
    )


def _run_episode_in_process_worker(
    payload: tuple[int, str, str, int, bool, int],
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int, int]:
    global _WORKER_MCTS
    if _WORKER_MCTS is None:
        raise RuntimeError("Worker MCTS is not initialized.")
    worker_mcts = cast("MCTS", _WORKER_MCTS)
    episode_seed, opponent_type, heuristic_level, model_player, add_noise, temp_threshold = (
        payload
    )
    rng = np.random.default_rng(seed=episode_seed)
    return play_episode(
        mcts=worker_mcts,
        add_noise=add_noise,
        temp_threshold=temp_threshold,
        rng=rng,
        opponent_type=opponent_type,
        opponent_heuristic_level=heuristic_level,
        model_player=model_player,
    )


def update_stats(stats: dict[str, float | int], winner: int, turn_idx: int) -> None:
    stats["total_turns"] = int(stats["total_turns"]) + turn_idx
    if winner == 1:
        stats["wins_p1"] = int(stats["wins_p1"]) + 1
        return
    if winner == -1:
        stats["wins_p2"] = int(stats["wins_p2"]) + 1
        return
    stats["draws"] = int(stats["draws"]) + 1


def history_to_examples(
    game_history: list[tuple[np.ndarray, np.ndarray, int]],
    winner: int,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for observation, policy, player_at_turn in game_history:
        if winner == 0:
            z = 0.0
        elif winner == player_at_turn:
            z = 1.0
        else:
            z = -1.0
        examples.append((observation, policy, z))
    return examples


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


def execute_self_play(
    system: AtaxxZero,
    buffer: ReplayBuffer,
    iteration: int,
    device: str,
) -> dict[str, float | int]:
    from engine.mcts import MCTS
    from training.curriculum import get_curriculum_mix, sample_opponent_from_curriculum

    system.eval()
    system.to(device)
    mcts = MCTS(
        model=system.model,
        c_puct=cfg_float("c_puct"),
        n_simulations=cfg_int("mcts_sims"),
        device=device,
        use_amp=cfg_bool("mcts_use_amp"),
        cache_size=max(0, cfg_int("mcts_cache_size")),
        leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
    )

    episodes = cfg_int("episodes_per_iter")
    temp_threshold = cfg_int("temp_threshold")
    add_noise = cfg_bool("add_noise")
    rng = np.random.default_rng(seed=cfg_int("seed") + iteration)
    selfplay_workers = cfg_int("selfplay_workers")
    log(f"[Iteration {iteration}] Self-play episodes: {episodes}", verbose_only=True)
    curriculum_mix = get_curriculum_mix(iteration)
    log(
        "  Opponent mix: "
        f"self={curriculum_mix['self']:.2f}, "
        f"heuristic={curriculum_mix['heuristic']:.2f}, "
        f"random={curriculum_mix['random']:.2f}",
        verbose_only=True,
    )
    log(
        "  Heuristic levels: "
        f"easy={curriculum_mix['heu_easy']:.2f}, "
        f"normal={curriculum_mix['heu_normal']:.2f}, "
        f"hard={curriculum_mix['heu_hard']:.2f}",
        verbose_only=True,
    )

    stats: dict[str, float | int] = {
        "wins_p1": 0,
        "wins_p2": 0,
        "draws": 0,
        "total_turns": 0,
        "avg_game_length": 0.0,
        "episodes_vs_self": 0,
        "episodes_vs_heuristic": 0,
        "episodes_vs_random": 0,
        "episodes_vs_heuristic_easy": 0,
        "episodes_vs_heuristic_normal": 0,
        "episodes_vs_heuristic_hard": 0,
    }

    episode_specs: list[tuple[int, str, str, int]] = []
    for episode_idx in range(episodes):
        opponent_type, heuristic_level = sample_opponent_from_curriculum(
            rng=rng,
            iteration=iteration,
        )
        model_player = 1 if float(rng.random()) >= cfg_float("model_side_swap_prob") else -1
        episode_seed = cfg_int("seed") + iteration * 10_000 + episode_idx
        episode_specs.append((episode_seed, opponent_type, heuristic_level, model_player))

    used_parallel = False
    episode_results: list[tuple[str, str, list[tuple[np.ndarray, np.ndarray, int]], int, int]] = []

    if selfplay_workers > 1:
        try:
            max_workers = min(selfplay_workers, episodes)
            worker_payloads = [
                (
                    episode_seed,
                    opponent_type,
                    heuristic_level,
                    model_player,
                    add_noise,
                    temp_threshold,
                )
                for episode_seed, opponent_type, heuristic_level, model_player in episode_specs
            ]
            model_state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in system.model.state_dict().items()
            }
            model_cfg: dict[str, int | float] = {
                "d_model": cfg_int("d_model"),
                "nhead": cfg_int("nhead"),
                "num_layers": cfg_int("num_layers"),
                "dim_feedforward": cfg_int("dim_feedforward"),
                "dropout": cfg_float("dropout"),
            }
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_selfplay_process_worker,
                initargs=(
                    model_state_dict,
                    model_cfg,
                    cfg_float("c_puct"),
                    cfg_int("mcts_sims"),
                ),
            ) as executor:
                for (
                    (_, opponent_type, heuristic_level, _),
                    episode_result,
                ) in zip(
                    episode_specs,
                    executor.map(_run_episode_in_process_worker, worker_payloads),
                    strict=True,
                ):
                    game_history, winner, turn_idx = episode_result
                    episode_results.append(
                        (opponent_type, heuristic_level, game_history, winner, turn_idx)
                    )
            used_parallel = True
            log(f"  Self-play process workers active: {max_workers}", verbose_only=True)
        except Exception as exc:
            log(
                f"  Process self-play failed, falling back to sequential mode: {exc}",
            )
            episode_results.clear()

    if not used_parallel:
        for episode_seed, opponent_type, heuristic_level, model_player in episode_specs:
            local_rng = np.random.default_rng(seed=episode_seed)
            game_history, winner, turn_idx = play_episode(
                mcts=mcts,
                add_noise=add_noise,
                temp_threshold=temp_threshold,
                rng=local_rng,
                opponent_type=opponent_type,
                opponent_heuristic_level=heuristic_level,
                model_player=model_player,
            )
            episode_results.append((opponent_type, heuristic_level, game_history, winner, turn_idx))

    for episode_idx, (opponent_type, heuristic_level, game_history, winner, turn_idx) in enumerate(
        episode_results,
        start=1,
    ):
        stats[f"episodes_vs_{opponent_type}"] = int(stats[f"episodes_vs_{opponent_type}"]) + 1
        if opponent_type == "heuristic":
            stats[f"episodes_vs_heuristic_{heuristic_level}"] = int(
                stats[f"episodes_vs_heuristic_{heuristic_level}"]
            ) + 1
        update_stats(stats=stats, winner=winner, turn_idx=turn_idx)
        buffer.save_game(history_to_examples(game_history=game_history, winner=winner))

        log_every = cfg_int("episode_log_every")
        if log_every > 0 and episode_idx % log_every == 0:
            log(
                f"  Episode {episode_idx}/{episodes} | winner={winner} turns={turn_idx}",
                verbose_only=True,
            )

    stats["avg_game_length"] = float(stats["total_turns"]) / float(episodes)
    cache_stats = mcts.cache_stats()
    stats["cache_hits"] = int(cache_stats["hits"])
    stats["cache_misses"] = int(cache_stats["misses"])
    stats["cache_hit_rate"] = float(cache_stats["hit_rate"])
    log(
        "  Self-play summary: "
        f"P1={stats['wins_p1']} P2={stats['wins_p2']} draws={stats['draws']} "
        f"avg_turns={stats['avg_game_length']:.1f} "
        f"cache_hit={float(stats['cache_hit_rate']):.1%}",
        verbose_only=True,
    )
    return stats


__all__ = [
    "evaluate_model",
    "execute_self_play",
]
