from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

root = Path(__file__).resolve().parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
from inference.checkpoint_duel_runtime import run_match_results_to_summary  # noqa: E402
from training.callbacks import OptimizerStateTransfer  # noqa: E402
from training.checkpointing import (  # noqa: E402
    cleanup_local_checkpoints,
    cleanup_old_log_versions,
    drain_completed_hf_uploads,
    ensure_hf_ready,
    init_hf_checkpointer,
    resolve_hf_start_iteration,
    should_save_iteration_checkpoint,
    wait_for_hf_uploads,
)
from training.config_runtime import (  # noqa: E402
    CONFIG,
    apply_cli_overrides,
    cfg_bool,
    cfg_float,
    cfg_int,
    cfg_str,
    ensure_src_on_path,
    log,
    parse_args,
    validate_config,
)
from training.eval_gating import compute_regression_gate  # noqa: E402
from training.eval_runtime import evaluate_model  # noqa: E402
from training.league_runtime import (  # noqa: E402
    record_checkpoint_in_league,
    resolve_champion_entry,
)
from training.logging_runtime import build_training_logger  # noqa: E402
from training.loop_runtime import (  # noqa: E402
    build_train_loader,
    build_val_loader,
    fit_with_ddp_fallback,
    prepare_train_val_examples,
    resolve_eval_levels,
    restore_system_from_checkpoint,
    run_warmup_if_needed,
)
from training.monitor import TrainingMonitor  # noqa: E402
from training.progress_callbacks import EpochPulseCallback  # noqa: E402
from training.selfplay_runtime import execute_self_play  # noqa: E402
from training.trainer_runtime import (  # noqa: E402
    export_onnx,
    resolve_trainer_hw,
    resolve_trainer_precision,
)


def main() -> None:
    args = parse_args()
    apply_cli_overrides(args)
    validate_config()
    ensure_src_on_path()

    from data.replay_buffer import ReplayBuffer
    from model.system import AtaxxZero

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    trainer_accelerator, trainer_devices, trainer_strategy = resolve_trainer_hw()
    device = "cuda" if trainer_accelerator == "gpu" else "cpu"
    log(f"Device: {device}")
    trainer_precision = resolve_trainer_precision(trainer_accelerator)
    log(
        "Trainer HW: "
        f"accelerator={trainer_accelerator}, devices={trainer_devices}, strategy={trainer_strategy}",
    )
    log(f"Trainer precision: {trainer_precision}")

    checkpoint_dir = Path(cfg_str("checkpoint_dir"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg_str("log_dir"))
    log_dir.mkdir(parents=True, exist_ok=True)

    iterations = cfg_int("iterations")
    epochs = cfg_int("epochs")
    system = AtaxxZero(
        learning_rate=cfg_float("learning_rate"),
        weight_decay=cfg_float("weight_decay"),
        value_loss_coeff=cfg_float("value_loss_coeff"),
        d_model=cfg_int("d_model"),
        nhead=cfg_int("nhead"),
        num_layers=cfg_int("num_layers"),
        dim_feedforward=cfg_int("dim_feedforward"),
        dropout=cfg_float("dropout"),
        scheduler_type="cosine",
        max_epochs=iterations * epochs,
    )
    if device == "cuda" and cfg_bool("compile_model"):
        try:
            system.model = cast(Any, torch.compile(system.model, mode="reduce-overhead"))
            log("Model compile enabled: torch.compile(mode='reduce-overhead').")
        except Exception as exc:
            log(f"Model compile skipped due to runtime error: {exc}")
    buffer = ReplayBuffer(capacity=cfg_int("buffer_size"))
    hf_checkpointer = init_hf_checkpointer()
    ensure_hf_ready(hf_checkpointer)
    hf_upload_executor: ThreadPoolExecutor | None = None
    hf_upload_futures: list[Future[None]] = []
    if hf_checkpointer is not None:
        hf_upload_executor = ThreadPoolExecutor(max_workers=1)
        try:
            start_iteration = resolve_hf_start_iteration(
                checkpointer=hf_checkpointer,
                system=system,
                buffer=buffer,
            )
        except (ValueError, OSError):
            start_iteration = 0
            log("HF resume failed; starting from scratch.")
    else:
        start_iteration = 0

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="ataxx-epoch{epoch:02d}-val{val_loss:.3f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = build_training_logger(log_dir)
    best_eval_score = -1.0
    best_path = checkpoint_dir / "best_eval.ckpt"
    eval_regression_streak = 0
    optimizer_transfer = OptimizerStateTransfer()
    monitor = TrainingMonitor(
        total_iterations=iterations,
        log_every=cfg_int("monitor_log_every"),
    )
    epoch_pulse = EpochPulseCallback(
        monitor=monitor,
        pulse_every=cfg_int("epoch_pulse_every"),
    )

    trainer_accelerator, trainer_devices, trainer_strategy, trainer_precision = run_warmup_if_needed(
        start_iteration=start_iteration,
        system=system,
        buffer=buffer,
        trainer_accelerator=trainer_accelerator,
        trainer_devices=trainer_devices,
        trainer_strategy=trainer_strategy,
        trainer_precision=trainer_precision,
        checkpoint_callback=checkpoint_callback,
        lr_monitor=lr_monitor,
        logger=logger,
        device=device,
        optimizer_transfer=optimizer_transfer,
        monitor=monitor,
        epoch_pulse=epoch_pulse,
    )

    try:
        for iteration in range(start_iteration + 1, iterations + 1):
            if hf_upload_futures:
                hf_upload_futures = drain_completed_hf_uploads(hf_upload_futures, fail_on_error=cfg_bool("fail_on_hf_upload_error"))
                if len(hf_upload_futures) > cfg_int("max_pending_hf_uploads"):
                    raise RuntimeError("HF upload backlog is growing; aborting early.")
            epoch_pulse.set_iteration(iteration)
            selfplay_start = time.perf_counter()
            selfplay_stats = execute_self_play(
                system=system,
                buffer=buffer,
                iteration=iteration,
                device=device,
            )
            selfplay_s = time.perf_counter() - selfplay_start
            if len(buffer) == 0:
                raise RuntimeError("Replay buffer is empty after self-play; aborting early.")

            train_examples, val_examples = prepare_train_val_examples(
                buffer=buffer,
                split_seed=cfg_int("seed") + iteration,
            )
            train_loader = build_train_loader(train_examples, device=device)
            val_loader = build_val_loader(val_examples, device=device)

            fit_start = time.perf_counter()
            trainer, trainer_accelerator, trainer_devices, trainer_strategy, trainer_precision = (
                fit_with_ddp_fallback(
                    system=system,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    trainer_accelerator=trainer_accelerator,
                    trainer_devices=trainer_devices,
                    trainer_strategy=trainer_strategy,
                    trainer_precision=trainer_precision,
                    checkpoint_callback=checkpoint_callback,
                    lr_monitor=lr_monitor,
                    logger=logger,
                    optimizer_transfer=optimizer_transfer,
                    epoch_pulse=epoch_pulse,
                )
            )
            fit_s = time.perf_counter() - fit_start
            monitor.log_iteration(
                iteration=iteration,
                selfplay_s=selfplay_s,
                fit_s=fit_s,
                buffer_size=len(buffer),
                selfplay_stats=selfplay_stats,
                logged_metrics=trainer.logged_metrics,
            )

            eval_stats: dict[str, float | int | str] | None = None
            eval_level_summaries: dict[str, dict[str, float | int | str]] = {}
            if cfg_bool("eval_enabled") and iteration % cfg_int("eval_every") == 0:
                try:
                    eval_levels = resolve_eval_levels()
                    level_scores: dict[str, float] = {}
                    eval_score_wins = 0
                    eval_score_losses = 0
                    eval_score_draws = 0
                    for level_idx, heuristic_level in enumerate(eval_levels):
                        current_eval = evaluate_model(
                            system=system,
                            device=device,
                            games=cfg_int("eval_games"),
                            sims=cfg_int("eval_sims"),
                            c_puct=cfg_float("c_puct"),
                            heuristic_level=heuristic_level,
                            seed=cfg_int("seed") + 10_000 + iteration + (level_idx * 997),
                        )
                        monitor.log_eval_snapshot(iteration=iteration, eval_stats=current_eval)
                        eval_level_summaries[heuristic_level] = current_eval
                        level_scores[heuristic_level] = float(current_eval["score"])
                        eval_score_wins += int(current_eval["wins"])
                        eval_score_losses += int(current_eval["losses"])
                        eval_score_draws += int(current_eval["draws"])

                    is_best = monitor.log_eval_composite(
                        iteration=iteration,
                        level_scores=level_scores,
                    )
                    eval_stats = {
                        "score": float(sum(level_scores.values()) / max(1, len(level_scores))),
                        "eval_total_wins": eval_score_wins,
                        "eval_total_losses": eval_score_losses,
                        "eval_total_draws": eval_score_draws,
                        "eval_levels": ",".join(level_scores.keys()),
                        **{
                            f"eval_score_{level}": score
                            for level, score in level_scores.items()
                        },
                    }
                    if is_best:
                        best_eval_score = float(eval_stats["score"])
                        trainer.save_checkpoint(str(best_path))
                        eval_regression_streak = 0
                    else:
                        eval_regression_streak, should_restore = compute_regression_gate(
                            current_score=float(eval_stats["score"]),
                            best_score=best_eval_score,
                            regression_delta=cfg_float("eval_regression_delta"),
                            current_streak=eval_regression_streak,
                            patience=cfg_int("eval_regression_patience"),
                        )
                        if (
                            cfg_bool("restore_best_on_regression")
                            and should_restore
                            and best_path.exists()
                        ):
                            try:
                                restore_system_from_checkpoint(system, str(best_path))
                                eval_regression_streak = 0
                                monitor.log_warning(
                                    iteration=iteration,
                                    message=(
                                        "eval regression detected; restored model weights from "
                                        f"{best_path.name}"
                                    ),
                                )
                            except Exception as restore_exc:
                                monitor.log_warning(
                                    iteration=iteration,
                                    message=f"failed to restore best checkpoint: {restore_exc}",
                                )
                except Exception as exc:
                    monitor.log_warning(iteration=iteration, message=f"eval failed, continuing training: {exc}")

            if not should_save_iteration_checkpoint(
                iteration=iteration,
                total_iterations=iterations,
                save_every=cfg_int("save_every"),
            ):
                continue

            manual_ckpt = checkpoint_dir / f"manual_iter_{iteration:03d}.ckpt"
            try:
                trainer.save_checkpoint(str(manual_ckpt))
                monitor.log_checkpoint(iteration=iteration, path=str(manual_ckpt))
                cleanup_local_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    keep_last_n=cfg_int("keep_last_n_local_checkpoints"),
                )
            except OSError:
                monitor.log_warning(iteration=iteration, message="local checkpoint save failed.")

            if cfg_bool("league_enabled") and len(eval_level_summaries) > 0:
                try:
                    champion_entry = resolve_champion_entry(current_checkpoint_path=manual_ckpt)
                    champion_series_summary: dict[str, float | int | str] | None = None
                    if champion_entry is not None and cfg_int("league_champion_games") > 0:
                        champion_series_summary = cast(
                            dict[str, float | int | str],
                            run_match_results_to_summary(
                                checkpoint_a=manual_ckpt,
                                checkpoint_b=champion_entry.artifact_path,
                                games=cfg_int("league_champion_games"),
                                device=device,
                                mcts_sims=cfg_int("eval_sims"),
                                c_puct=cfg_float("c_puct"),
                                seed=cfg_int("seed") + 200_000 + iteration,
                            ),
                        )
                        monitor.log_warning(
                            iteration=iteration,
                            message=(
                                "league champion duel "
                                f"score={float(champion_series_summary['checkpoint_a_score']):.3f} "
                                f"vs {champion_entry.display_name}"
                            ),
                        )
                    updated_league = record_checkpoint_in_league(
                        checkpoint_path=manual_ckpt,
                        heuristic_series_by_level=eval_level_summaries,
                        champion_entry=champion_entry,
                        champion_series_summary=champion_series_summary,
                    )
                    monitor.log_warning(
                        iteration=iteration,
                        message=f"league updated champion={updated_league.get('champion_id')}",
                    )
                except Exception as exc:
                    monitor.log_warning(
                        iteration=iteration,
                        message=f"league update failed, continuing training: {exc}",
                    )

            if hf_checkpointer is not None:
                try:
                    model_path, buffer_path, metadata_path = hf_checkpointer.save_checkpoint_local(
                        iteration=iteration,
                        system=system,
                        buffer=buffer,
                        config=CONFIG,
                        stats={
                            "replay_size": len(buffer),
                            "best_eval_score": best_eval_score,
                            **(eval_stats or {}),
                        },
                    )
                    if hf_upload_executor is not None:
                        future = hf_upload_executor.submit(
                            hf_checkpointer.upload_checkpoint_files,
                            iteration=iteration,
                            model_path=model_path,
                            buffer_path=buffer_path,
                            metadata_path=metadata_path,
                            keep_last_n=cfg_int("keep_last_n_hf_checkpoints"),
                        )
                        hf_upload_futures.append(future)
                        monitor.log_warning(iteration=iteration, message=f"HF upload queued for iteration {iteration}.")
                        if len(hf_upload_futures) > cfg_int("max_pending_hf_uploads"):
                            raise RuntimeError("HF upload backlog exceeded configured threshold.")
                    else:
                        hf_checkpointer.upload_checkpoint_files(
                            iteration=iteration,
                            model_path=model_path,
                            buffer_path=buffer_path,
                            metadata_path=metadata_path,
                            keep_last_n=cfg_int("keep_last_n_hf_checkpoints"),
                        )
                        monitor.log_warning(iteration=iteration, message=f"HF checkpoint uploaded for iteration {iteration}.")
                except (OSError, ValueError):
                    monitor.log_warning(iteration=iteration, message="HF upload failed for this iteration.")
            if cfg_bool("export_onnx"):
                try:
                    export_onnx(system.model, cfg_str("onnx_path"), device=device)
                except (OSError, RuntimeError, ValueError):
                    monitor.log_warning(iteration=iteration, message="ONNX export failed for this iteration.")

            cleanup_old_log_versions(
                log_dir=log_dir,
                run_name="ataxx_zero",
                keep_last_n=cfg_int("keep_last_n_log_versions"),
            )
    finally:
        if hf_upload_executor is not None:
            try:
                wait_for_hf_uploads(
                    hf_upload_futures,
                    timeout_s=cfg_float("hf_upload_future_timeout_s"),
                    fail_on_error=cfg_bool("fail_on_hf_upload_error"),
                )
            except Exception as exc:
                if cfg_bool("fail_on_hf_upload_error"):
                    raise
                log(f"HF upload wait failed: {exc}")
            hf_upload_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
