from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from agents.heuristic import heuristic_move
from game.actions import ACTION_SPACE
from training.config_runtime import cfg_bool, cfg_float, cfg_int, log
from training.reward_runtime import (
    HistoryEntry,
    compute_state_potential,
    compute_transition_shaping_reward,
)

if TYPE_CHECKING:
    from engine.mcts import MCTS, MCTSNode
    from game.board import AtaxxBoard
    from training.league_runtime import CheckpointModelSpec

_WORKER_MCTS: object | None = None
_WORKER_OPPONENT_MCTS_POOL: dict[str, object] | None = None


def _one_hot_policy(action_idx: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
    policy[action_idx] = 1.0
    return policy


def _append_history_entry(
    *,
    game_history: list[HistoryEntry],
    board: AtaxxBoard,
    player_at_turn: int,
    policy: np.ndarray,
    action_idx: int,
) -> None:
    shaping_enabled = cfg_bool("reward_shaping_enabled")
    observation = board.get_observation()
    shaping_reward = 0.0
    before_potential = 0.0
    if shaping_enabled:
        before_potential = compute_state_potential(board, player_at_turn)
    board.step(ACTION_SPACE.decode(action_idx))
    if shaping_enabled:
        after_potential = compute_state_potential(board, player_at_turn)
        shaping_reward = compute_transition_shaping_reward(
            before_potential=before_potential,
            after_potential=after_potential,
        )
    game_history.append((observation, policy, player_at_turn, shaping_reward))


def _resolve_selfplay_worker_device(worker_devices: tuple[str, ...]) -> str:
    if len(worker_devices) == 0:
        return "cpu"
    if len(worker_devices) == 1:
        return worker_devices[0]

    identity = getattr(mp.current_process(), "_identity", ())
    if len(identity) > 0 and isinstance(identity[0], int):
        slot = max(0, int(identity[0]) - 1)
        return worker_devices[slot % len(worker_devices)]

    name = mp.current_process().name
    suffix = name.rsplit("-", maxsplit=1)[-1]
    if suffix.isdigit():
        slot = max(0, int(suffix) - 1)
        return worker_devices[slot % len(worker_devices)]
    return worker_devices[0]


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
    opponent_checkpoint_mcts: MCTS | None = None,
) -> tuple[list[HistoryEntry], int, int, bool]:
    from game.board import AtaxxBoard

    board = AtaxxBoard()
    model_root = None
    opponent_root = None
    game_history: list[HistoryEntry] = []
    turn_idx = 0

    while not board.is_game_over():
        turn_idx += 1
        is_model_turn = board.current_player == model_player
        if is_model_turn:
            temperature = 1.0 if turn_idx <= temp_threshold else 0.0
            probs, model_root = compute_action_probs(
                board=board,
                mcts=mcts,
                root=model_root,
                add_noise=add_noise,
                temperature=temperature,
            )
            player_at_turn = int(board.current_player)
            action_idx = select_action_idx(probs=probs, temperature=temperature, rng=rng)
            policy = probs.astype(np.float32, copy=False)
            _append_history_entry(
                game_history=game_history,
                board=board,
                player_at_turn=player_at_turn,
                policy=policy,
                action_idx=action_idx,
            )
            model_root = mcts.advance_root(model_root, action_idx)
            if opponent_type == "checkpoint" and opponent_checkpoint_mcts is not None:
                opponent_root = opponent_checkpoint_mcts.advance_root(opponent_root, action_idx)
            continue

        if opponent_type == "self":
            temperature = 1.0 if turn_idx <= temp_threshold else 0.0
            probs, model_root = compute_action_probs(
                board=board,
                mcts=mcts,
                root=model_root,
                add_noise=add_noise,
                temperature=temperature,
            )
            player_at_turn = int(board.current_player)
            action_idx = select_action_idx(probs=probs, temperature=temperature, rng=rng)
            policy = probs.astype(np.float32, copy=False)
            _append_history_entry(
                game_history=game_history,
                board=board,
                player_at_turn=player_at_turn,
                policy=policy,
                action_idx=action_idx,
            )
            model_root = mcts.advance_root(model_root, action_idx)
            continue

        if opponent_type == "checkpoint":
            if opponent_checkpoint_mcts is None:
                raise RuntimeError("Checkpoint opponent selected without checkpoint MCTS.")
            temperature = 1.0 if turn_idx <= temp_threshold else 0.0
            probs, opponent_root = compute_action_probs(
                board=board,
                mcts=opponent_checkpoint_mcts,
                root=opponent_root,
                add_noise=False,
                temperature=temperature,
            )
            action_idx = select_action_idx(probs=probs, temperature=temperature, rng=rng)
            _append_history_entry(
                game_history=game_history,
                board=board,
                player_at_turn=int(board.current_player),
                policy=probs.astype(np.float32, copy=False),
                action_idx=action_idx,
            )
            model_root = mcts.advance_root(model_root, action_idx)
            opponent_root = opponent_checkpoint_mcts.advance_root(opponent_root, action_idx)
            continue

        if opponent_type == "heuristic":
            move = heuristic_move(board, rng, opponent_heuristic_level)
            action_idx = ACTION_SPACE.encode(move)
            _append_history_entry(
                game_history=game_history,
                board=board,
                player_at_turn=int(board.current_player),
                policy=_one_hot_policy(action_idx),
                action_idx=action_idx,
            )
            model_root = mcts.advance_root(model_root, action_idx)
            continue

        move = random_move(board, rng)
        board.step(move)
        model_root = mcts.advance_root(model_root, ACTION_SPACE.encode(move))

    return game_history, board.get_result(), turn_idx, board.is_forced_draw()


def build_sequential_checkpoint_mcts_pool(
    *,
    specs: dict[str, CheckpointModelSpec],
    device: str,
) -> dict[str, MCTS]:
    from engine.mcts import MCTS
    from model.transformer import AtaxxTransformerNet

    pool: dict[str, MCTS] = {}
    for participant_id, spec in specs.items():
        opponent_model = AtaxxTransformerNet(
            d_model=int(spec.model_cfg["d_model"]),
            nhead=int(spec.model_cfg["nhead"]),
            num_layers=int(spec.model_cfg["num_layers"]),
            dim_feedforward=int(spec.model_cfg["dim_feedforward"]),
            dropout=float(spec.model_cfg["dropout"]),
        )
        opponent_model.load_state_dict(spec.state_dict)
        opponent_model.eval()
        pool[participant_id] = MCTS(
            model=opponent_model,
            c_puct=cfg_float("c_puct"),
            n_simulations=cfg_int("mcts_sims"),
            device=device,
            use_amp=cfg_bool("mcts_use_amp"),
            cache_size=max(0, cfg_int("mcts_cache_size")),
            leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
        )
    return pool


def init_selfplay_process_worker(
    model_state_dict: dict[str, torch.Tensor],
    model_cfg: dict[str, int | float],
    c_puct: float,
    sims: int,
    opponent_specs: dict[str, tuple[dict[str, torch.Tensor], dict[str, int | float]]],
    worker_devices: tuple[str, ...],
) -> None:
    global _WORKER_MCTS
    global _WORKER_OPPONENT_MCTS_POOL
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
    worker_device = _resolve_selfplay_worker_device(worker_devices)
    use_amp = cfg_bool("mcts_use_amp") and worker_device.startswith("cuda")
    _WORKER_MCTS = MCTS(
        model=model,
        c_puct=c_puct,
        n_simulations=sims,
        device=worker_device,
        use_amp=use_amp,
        cache_size=max(0, cfg_int("mcts_cache_size")),
        leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
    )
    _WORKER_OPPONENT_MCTS_POOL = {}
    for participant_id, (opponent_state_dict, opponent_cfg) in opponent_specs.items():
        opponent_model = AtaxxTransformerNet(
            d_model=int(opponent_cfg["d_model"]),
            nhead=int(opponent_cfg["nhead"]),
            num_layers=int(opponent_cfg["num_layers"]),
            dim_feedforward=int(opponent_cfg["dim_feedforward"]),
            dropout=float(opponent_cfg["dropout"]),
        )
        opponent_model.load_state_dict(opponent_state_dict)
        opponent_model.eval()
        _WORKER_OPPONENT_MCTS_POOL[participant_id] = MCTS(
            model=opponent_model,
            c_puct=c_puct,
            n_simulations=sims,
            device=worker_device,
            use_amp=use_amp,
            cache_size=max(0, cfg_int("mcts_cache_size")),
            leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
        )


def run_episode_in_process_worker(
    payload: tuple[int, str, str, int, bool, int, str],
) -> tuple[list[HistoryEntry], int, int, bool]:
    global _WORKER_MCTS
    global _WORKER_OPPONENT_MCTS_POOL
    if _WORKER_MCTS is None:
        raise RuntimeError("Worker MCTS is not initialized.")
    worker_mcts = cast("MCTS", _WORKER_MCTS)
    episode_seed, opponent_type, heuristic_level, model_player, add_noise, temp_threshold, checkpoint_id = payload
    rng = np.random.default_rng(seed=episode_seed)
    opponent_checkpoint_mcts = None
    if opponent_type == "checkpoint":
        pool = _WORKER_OPPONENT_MCTS_POOL or {}
        opponent_checkpoint_mcts = cast("MCTS | None", pool.get(checkpoint_id))
    return play_episode(
        mcts=worker_mcts,
        add_noise=add_noise,
        temp_threshold=temp_threshold,
        rng=rng,
        opponent_type=opponent_type,
        opponent_heuristic_level=heuristic_level,
        model_player=model_player,
        opponent_checkpoint_mcts=opponent_checkpoint_mcts,
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


def handle_parallel_selfplay_failure(exc: Exception) -> None:
    if cfg_bool("fail_on_selfplay_parallel_error"):
        raise RuntimeError(
            "Process self-play failed with parallel workers. "
            "Aborting instead of silently falling back to sequential mode.",
        ) from exc
    log(f"  Process self-play failed, falling back to sequential mode: {exc}")


__all__ = [
    "build_sequential_checkpoint_mcts_pool",
    "compute_action_probs",
    "handle_parallel_selfplay_failure",
    "init_selfplay_process_worker",
    "play_episode",
    "run_episode_in_process_worker",
    "select_action_idx",
    "update_stats",
]
