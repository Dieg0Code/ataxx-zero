from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING

import numpy as np
import torch

from agents.heuristic import HEURISTIC_LEVELS
from training.config_runtime import cfg_bool, cfg_float, cfg_int, log
from training.league_runtime import (
    CheckpointModelSpec,
    CheckpointPoolEntry,
    build_checkpoint_pool_specs,
    select_checkpoint_pool_entry,
)
from training.reward_runtime import history_to_examples
from training.selfplay_episode_runtime import (
    build_sequential_checkpoint_mcts_pool,
    compute_action_probs,
    handle_parallel_selfplay_failure,
    init_selfplay_process_worker,
    play_episode,
    run_episode_in_process_worker,
    update_stats,
)

if TYPE_CHECKING:
    from data.replay_buffer import ReplayBuffer
    from model.system import AtaxxZero
    from training.reward_runtime import HistoryEntry


def _build_episode_specs(
    *,
    iteration: int,
    episodes: int,
    rng: np.random.Generator,
    opponent_pool_entries: list[CheckpointPoolEntry],
) -> list[tuple[int, str, str, int, str]]:
    from training.curriculum import sample_opponent_from_curriculum

    episode_specs: list[tuple[int, str, str, int, str]] = []
    for episode_idx in range(episodes):
        opponent_type, heuristic_level = sample_opponent_from_curriculum(
            rng=rng,
            iteration=iteration,
        )
        checkpoint_id = ""
        if (
            opponent_type == "self"
            and len(opponent_pool_entries) > 0
            and float(rng.random()) < cfg_float("league_selfplay_checkpoint_prob")
        ):
            selected_entry = select_checkpoint_pool_entry(
                entries=opponent_pool_entries,
                rng=rng,
            )
            if selected_entry is not None:
                opponent_type = "checkpoint"
                checkpoint_id = selected_entry.participant_id
        model_player = 1 if float(rng.random()) >= cfg_float("model_side_swap_prob") else -1
        episode_seed = cfg_int("seed") + iteration * 10_000 + episode_idx
        episode_specs.append(
            (episode_seed, opponent_type, heuristic_level, model_player, checkpoint_id),
        )
    return episode_specs


def _log_curriculum_mix(*, iteration: int) -> None:
    from training.curriculum import get_curriculum_mix

    curriculum_mix = get_curriculum_mix(iteration)
    log(
        f"[Iteration {iteration}] self-play start episodes={cfg_int('episodes_per_iter')} "
        f"sims={cfg_int('mcts_sims')} workers={cfg_int('selfplay_workers')}",
    )
    log(
        "  Opponent mix: "
        f"self={curriculum_mix['self']:.2f}, "
        f"heuristic={curriculum_mix['heuristic']:.2f}, "
        f"random={curriculum_mix['random']:.2f}",
        verbose_only=True,
    )
    heuristic_mix = " ".join(
        f"{level}={curriculum_mix[f'heu_{level}']:.2f}"
        for level in HEURISTIC_LEVELS
    )
    log(f"  Heuristic levels: {heuristic_mix}", verbose_only=True)


def _empty_stats() -> dict[str, float | int]:
    stats: dict[str, float | int] = {
        "wins_p1": 0,
        "wins_p2": 0,
        "draws": 0,
        "total_turns": 0,
        "avg_game_length": 0.0,
        "episodes_vs_self": 0,
        "episodes_vs_checkpoint": 0,
        "episodes_vs_heuristic": 0,
        "episodes_vs_random": 0,
    }
    for heuristic_level in HEURISTIC_LEVELS:
        stats[f"episodes_vs_heuristic_{heuristic_level}"] = 0
    return stats


def resolve_parallel_selfplay_worker_devices(
    *,
    device: str,
    requested_workers: int,
    episodes: int,
) -> list[str]:
    if requested_workers <= 1 or episodes <= 1:
        return []
    max_workers = min(requested_workers, episodes)
    if not device.startswith("cuda"):
        return ["cpu"] * max_workers
    if not torch.cuda.is_available():
        return ["cpu"] * max_workers
    gpu_count = max(0, int(torch.cuda.device_count()))
    # On a single GPU, multiprocessing only pushes MCTS off the training device and
    # costs more than it helps. We keep the sequential path on CUDA instead.
    if gpu_count <= 1:
        return []
    return [f"cuda:{idx}" for idx in range(min(max_workers, gpu_count))]


def _run_parallel_selfplay(
    *,
    iteration: int,
    system: AtaxxZero,
    episodes: int,
    episode_specs: list[tuple[int, str, str, int, str]],
    add_noise: bool,
    temp_threshold: int,
    progress_every_s: float,
    episode_timeout_s: float,
    selfplay_start: float,
    last_progress_log_s: float,
    opponent_pool_specs: dict[str, CheckpointModelSpec],
    worker_devices: list[str],
) -> tuple[list[tuple[str, str, list[HistoryEntry], int, int, bool]], float]:
    max_workers = len(worker_devices)
    worker_payloads = [
        (
            episode_seed,
            opponent_type,
            heuristic_level,
            model_player,
            add_noise,
            temp_threshold,
            checkpoint_id,
        )
        for episode_seed, opponent_type, heuristic_level, model_player, checkpoint_id in episode_specs
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
    serialized_opponent_specs = {
        participant_id: (spec.state_dict, spec.model_cfg)
        for participant_id, spec in opponent_pool_specs.items()
    }
    episode_results: list[tuple[str, str, list[HistoryEntry], int, int, bool]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_selfplay_process_worker,
        initargs=(
            model_state_dict,
            model_cfg,
            cfg_float("c_puct"),
            cfg_int("mcts_sims"),
            serialized_opponent_specs,
            tuple(worker_devices),
        ),
    ) as executor:
        futures: dict[Future[tuple[list[HistoryEntry], int, int, bool]], tuple[int, str, str]] = {}
        submitted_at: dict[Future[tuple[list[HistoryEntry], int, int, bool]], float] = {}
        ordered_results: list[tuple[int, str, str, list[HistoryEntry], int, int, bool]] = []
        for idx, ((_, opponent_type, heuristic_level, _, _), payload) in enumerate(
            zip(episode_specs, worker_payloads, strict=True),
            start=1,
        ):
            future = executor.submit(run_episode_in_process_worker, payload)
            futures[future] = (idx, opponent_type, heuristic_level)
            submitted_at[future] = time.perf_counter()

        pending = set(futures)
        while pending:
            done, pending = wait(
                pending,
                timeout=progress_every_s,
                return_when=FIRST_COMPLETED,
            )
            now_s = time.perf_counter()
            if not done:
                if (now_s - last_progress_log_s) >= progress_every_s:
                    log(
                        f"[Iteration {iteration}] self-play progress "  # type: ignore[name-defined]
                        f"{len(ordered_results)}/{episodes} elapsed={now_s - selfplay_start:.0f}s",
                    )
                    last_progress_log_s = now_s
                if episode_timeout_s <= 0.0:
                    continue
                oldest_pending_s = max(now_s - submitted_at[fut] for fut in pending)
                if oldest_pending_s <= episode_timeout_s:
                    continue
                raise TimeoutError(
                    "Parallel self-play stalled: "
                    f"oldest pending episode exceeded {episode_timeout_s:.0f}s.",
                )

            for future in done:
                idx, opponent_type, heuristic_level = futures[future]
                game_history, winner, turn_idx, forced_draw = future.result()
                ordered_results.append(
                    (idx, opponent_type, heuristic_level, game_history, winner, turn_idx, forced_draw),
                )
                submitted_at.pop(future, None)

            if (now_s - last_progress_log_s) >= progress_every_s:
                log(
                    f"[Iteration {iteration}] self-play progress "  # type: ignore[name-defined]
                    f"{len(ordered_results)}/{episodes} elapsed={now_s - selfplay_start:.0f}s",
                )
                last_progress_log_s = now_s

        ordered_results.sort(key=lambda item: item[0])
        for _idx, opponent_type, heuristic_level, game_history, winner, turn_idx, forced_draw in ordered_results:
            episode_results.append(
                (opponent_type, heuristic_level, game_history, winner, turn_idx, forced_draw),
            )
    log(f"[Iteration {iteration}] self-play process workers active: {max_workers}")  # type: ignore[name-defined]
    return episode_results, last_progress_log_s


def execute_self_play(
    system: AtaxxZero,
    buffer: ReplayBuffer,
    iteration: int,
    device: str,
) -> dict[str, float | int]:
    from engine.mcts import MCTS

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
    progress_every_s = max(5.0, cfg_float("selfplay_progress_every_s"))
    episode_timeout_s = max(0.0, cfg_float("selfplay_episode_timeout_s"))
    rng = np.random.default_rng(seed=cfg_int("seed") + iteration)
    selfplay_start = time.perf_counter()
    last_progress_log_s = selfplay_start

    _log_curriculum_mix(iteration=iteration)
    stats = _empty_stats()
    opponent_pool_specs = build_checkpoint_pool_specs(current_checkpoint_path=None)
    if len(opponent_pool_specs) > 0:
        pool_log = ", ".join(
            f"{spec.entry.display_name}:{spec.entry.rating:.0f}"
            for spec in opponent_pool_specs.values()
        )
        log(f"  Checkpoint pool: {pool_log}", verbose_only=True)

    episode_specs = _build_episode_specs(
        iteration=iteration,
        episodes=episodes,
        rng=rng,
        opponent_pool_entries=[spec.entry for spec in opponent_pool_specs.values()],
    )
    episode_results: list[tuple[str, str, list[HistoryEntry], int, int, bool]] = []
    parallel_worker_devices = resolve_parallel_selfplay_worker_devices(
        device=device,
        requested_workers=cfg_int("selfplay_workers"),
        episodes=episodes,
    )

    if (
        cfg_int("selfplay_workers") > 1
        and len(parallel_worker_devices) == 0
        and device.startswith("cuda")
    ):
        log(
            "  Parallel self-play disabled on single-GPU CUDA runtime; keeping MCTS on GPU.",
            verbose_only=True,
        )

    if len(parallel_worker_devices) > 0:
        try:
            episode_results, last_progress_log_s = _run_parallel_selfplay(
                iteration=iteration,
                system=system,
                episodes=episodes,
                episode_specs=episode_specs,
                add_noise=add_noise,
                temp_threshold=temp_threshold,
                progress_every_s=progress_every_s,
                episode_timeout_s=episode_timeout_s,
                selfplay_start=selfplay_start,
                last_progress_log_s=last_progress_log_s,
                opponent_pool_specs=opponent_pool_specs,
                worker_devices=parallel_worker_devices,
            )
        except Exception as exc:
            handle_parallel_selfplay_failure(exc)
            episode_results.clear()

    if len(episode_results) == 0:
        opponent_mcts_pool = build_sequential_checkpoint_mcts_pool(
            specs=opponent_pool_specs,
            device=device,
        )
        for episode_seed, opponent_type, heuristic_level, model_player, checkpoint_id in episode_specs:
            game_history, winner, turn_idx, forced_draw = play_episode(
                mcts=mcts,
                add_noise=add_noise,
                temp_threshold=temp_threshold,
                rng=np.random.default_rng(seed=episode_seed),
                opponent_type=opponent_type,
                opponent_heuristic_level=heuristic_level,
                model_player=model_player,
                opponent_checkpoint_mcts=opponent_mcts_pool.get(checkpoint_id),
            )
            episode_results.append(
                (opponent_type, heuristic_level, game_history, winner, turn_idx, forced_draw),
            )
            now_s = time.perf_counter()
            if (now_s - last_progress_log_s) >= progress_every_s:
                log(
                    f"[Iteration {iteration}] self-play progress "
                    f"{len(episode_results)}/{episodes} elapsed={now_s - selfplay_start:.0f}s",
                )
                last_progress_log_s = now_s

    for episode_idx, (
        opponent_type,
        heuristic_level,
        game_history,
        winner,
        turn_idx,
        forced_draw,
    ) in enumerate(episode_results, start=1):
        stats[f"episodes_vs_{opponent_type}"] = int(stats[f"episodes_vs_{opponent_type}"]) + 1
        if opponent_type == "heuristic":
            stats[f"episodes_vs_heuristic_{heuristic_level}"] = int(
                stats[f"episodes_vs_heuristic_{heuristic_level}"],
            ) + 1
        update_stats(stats=stats, winner=winner, turn_idx=turn_idx)
        buffer.save_game(
            history_to_examples(
                game_history=game_history,
                winner=winner,
                forced_draw=forced_draw,
            ),
        )

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
        f"  Self-play summary: P1={stats['wins_p1']} P2={stats['wins_p2']} draws={stats['draws']} "
        f"avg_turns={stats['avg_game_length']:.1f} cache_hit={float(stats['cache_hit_rate']):.1%}",
        verbose_only=True,
    )
    return stats


__all__ = [
    "compute_action_probs",
    "execute_self_play",
    "handle_parallel_selfplay_failure",
    "resolve_parallel_selfplay_worker_devices",
]
