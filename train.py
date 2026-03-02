from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

if TYPE_CHECKING:
    from data.replay_buffer import ReplayBuffer
    from model.system import AtaxxZero

from training.bootstrap import generate_imitation_data  # noqa: E402
from training.callbacks import OptimizerStateTransfer  # noqa: E402
from training.checkpointing import (  # noqa: E402
    cleanup_local_checkpoints,
    cleanup_old_log_versions,
    init_hf_checkpointer,
    should_save_iteration_checkpoint,
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
from training.eval_runtime import evaluate_model  # noqa: E402
from training.monitor import TrainingMonitor  # noqa: E402
from training.progress_callbacks import EpochPulseCallback  # noqa: E402
from training.selfplay_runtime import execute_self_play  # noqa: E402
from training.trainer_runtime import (  # noqa: E402
    build_trainer,
    export_onnx,
    is_ddp_rendezvous_timeout,
    resolve_trainer_hw,
    resolve_trainer_precision,
)


def _build_train_loader(buffer: ReplayBuffer, device: str) -> DataLoader[object]:
    from data.dataset import AtaxxDataset

    dataset = AtaxxDataset(
        buffer=buffer,
        augment=True,
        reference_buffer=False,
        val_split=cfg_float("val_split"),
    )
    loader_kwargs: dict[str, object] = {}
    if cfg_int("num_workers") > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=cfg_int("batch_size"),
        shuffle=True,
        num_workers=cfg_int("num_workers"),
        persistent_workers=(cfg_bool("persistent_workers") and cfg_int("num_workers") > 0),
        pin_memory=(device == "cuda"),
        **loader_kwargs,
    )


def _build_val_loader(buffer: ReplayBuffer, device: str) -> DataLoader[object] | None:
    from data.dataset import ValidationDataset

    val_dataset = ValidationDataset(buffer=buffer, split=cfg_float("val_split"))
    if len(val_dataset) == 0:
        return None
    loader_kwargs: dict[str, object] = {}
    if cfg_int("num_workers") > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        val_dataset,
        batch_size=cfg_int("batch_size"),
        shuffle=False,
        num_workers=cfg_int("num_workers"),
        persistent_workers=(cfg_bool("persistent_workers") and cfg_int("num_workers") > 0),
        pin_memory=(device == "cuda"),
        **loader_kwargs,
    )


def _fit_with_ddp_fallback(
    *,
    system: AtaxxZero,
    train_loader: DataLoader[object],
    val_loader: DataLoader[object] | None,
    epochs: int,
    trainer_accelerator: str,
    trainer_devices: int,
    trainer_strategy: str,
    trainer_precision: str,
    checkpoint_callback: ModelCheckpoint,
    lr_monitor: LearningRateMonitor,
    logger: TensorBoardLogger,
    optimizer_transfer: OptimizerStateTransfer,
    epoch_pulse: EpochPulseCallback,
) -> tuple[pl.Trainer, str, int, str, str]:
    trainer = build_trainer(
        epochs=epochs,
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        strategy=trainer_strategy,
        precision=trainer_precision,
        benchmark=cfg_bool("trainer_benchmark"),
        checkpoint_callback=checkpoint_callback,
        lr_monitor=lr_monitor,
        logger=logger,
        extra_callbacks=[optimizer_transfer, epoch_pulse],
    )
    system.train()
    try:
        trainer.fit(
            model=system,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    except Exception as exc:
        # DDP startup failures are environmental; we degrade to single-device
        # so long runs are not lost due to rendezvous flakiness.
        if trainer_devices <= 1 or not is_ddp_rendezvous_timeout(exc):
            raise
        log("DDP rendezvous failed. Falling back to single-GPU for this run.")
        trainer_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer_devices = 1
        trainer_strategy = "auto"
        trainer_precision = resolve_trainer_precision(trainer_accelerator)
        trainer = build_trainer(
            epochs=epochs,
            accelerator=trainer_accelerator,
            devices=trainer_devices,
            strategy=trainer_strategy,
            precision=trainer_precision,
            benchmark=cfg_bool("trainer_benchmark"),
            checkpoint_callback=checkpoint_callback,
            lr_monitor=lr_monitor,
            logger=logger,
            extra_callbacks=[optimizer_transfer, epoch_pulse],
        )
        system.train()
        trainer.fit(
            model=system,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    return (
        trainer,
        trainer_accelerator,
        trainer_devices,
        trainer_strategy,
        trainer_precision,
    )


def _run_warmup_if_needed(
    *,
    start_iteration: int,
    system: AtaxxZero,
    buffer: ReplayBuffer,
    trainer_accelerator: str,
    trainer_devices: int,
    trainer_strategy: str,
    trainer_precision: str,
    checkpoint_callback: ModelCheckpoint,
    lr_monitor: LearningRateMonitor,
    logger: TensorBoardLogger,
    device: str,
    optimizer_transfer: OptimizerStateTransfer,
    monitor: TrainingMonitor,
    epoch_pulse: EpochPulseCallback,
) -> None:
    warmup_games = cfg_int("warmup_games")
    warmup_epochs = cfg_int("warmup_epochs")
    if start_iteration != 0 or warmup_games <= 0 or warmup_epochs <= 0:
        return

    # Warmup seeds the policy with legal, sensible moves before self-play noise.
    warmup_rng = torch.Generator().manual_seed(cfg_int("seed"))
    rng_seed = int(torch.randint(0, 2**31, (1,), generator=warmup_rng).item())
    warmup_examples = generate_imitation_data(
        n_games=warmup_games,
        rng=np.random.default_rng(seed=rng_seed),
        heuristic_level="hard",
    )
    buffer.save_game(warmup_examples)
    monitor.log_warmup(examples=len(warmup_examples), games=warmup_games)
    train_loader = _build_train_loader(buffer, device=device)
    val_loader = _build_val_loader(buffer, device=device)
    warmup_trainer = build_trainer(
        epochs=warmup_epochs,
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        strategy=trainer_strategy,
        precision=trainer_precision,
        benchmark=cfg_bool("trainer_benchmark"),
        checkpoint_callback=checkpoint_callback,
        lr_monitor=lr_monitor,
        logger=logger,
        extra_callbacks=[optimizer_transfer, epoch_pulse],
    )
    system.train()
    warmup_trainer.fit(
        model=system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    trainer_accelerator, trainer_devices, trainer_strategy = resolve_trainer_hw()
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
            system.model = torch.compile(system.model, mode="reduce-overhead")
            log("Model compile enabled: torch.compile(mode='reduce-overhead').")
        except Exception as exc:
            log(f"Model compile skipped due to runtime error: {exc}")
    buffer = ReplayBuffer(capacity=cfg_int("buffer_size"))

    hf_checkpointer = init_hf_checkpointer()
    hf_upload_executor: ThreadPoolExecutor | None = None
    hf_upload_futures: list[object] = []
    if hf_checkpointer is not None:
        hf_upload_executor = ThreadPoolExecutor(max_workers=1)
        try:
            start_iteration = hf_checkpointer.load_latest_checkpoint(
                system=system,
                buffer=buffer,
            )
            log(f"Resumed from HF checkpoint iteration {start_iteration}.")
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
    logger = TensorBoardLogger(save_dir=str(log_dir), name="ataxx_zero")
    best_eval_score = -1.0
    optimizer_transfer = OptimizerStateTransfer()
    monitor = TrainingMonitor(
        total_iterations=iterations,
        log_every=cfg_int("monitor_log_every"),
    )
    epoch_pulse = EpochPulseCallback(
        monitor=monitor,
        pulse_every=cfg_int("epoch_pulse_every"),
    )

    _run_warmup_if_needed(
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
            epoch_pulse.set_iteration(iteration)
            selfplay_start = time.perf_counter()
            selfplay_stats = execute_self_play(
                system=system,
                buffer=buffer,
                iteration=iteration,
                device=device,
            )
            selfplay_s = time.perf_counter() - selfplay_start

            train_loader = _build_train_loader(buffer, device=device)
            val_loader = _build_val_loader(buffer, device=device)

            fit_start = time.perf_counter()
            trainer, trainer_accelerator, trainer_devices, trainer_strategy, trainer_precision = (
                _fit_with_ddp_fallback(
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
            if cfg_bool("eval_enabled") and iteration % cfg_int("eval_every") == 0:
                try:
                    current_eval = evaluate_model(
                        system=system,
                        device=device,
                        games=cfg_int("eval_games"),
                        sims=cfg_int("eval_sims"),
                        c_puct=cfg_float("c_puct"),
                        heuristic_level=cfg_str("eval_heuristic_level"),
                        seed=cfg_int("seed") + 10_000 + iteration,
                    )
                    eval_stats = current_eval
                    is_best = monitor.log_eval(iteration=iteration, eval_stats=current_eval)
                    if is_best:
                        best_eval_score = float(current_eval["score"])
                        best_path = checkpoint_dir / "best_eval.ckpt"
                        trainer.save_checkpoint(str(best_path))
                except Exception as exc:
                    monitor.log_warning(
                        iteration=iteration,
                        message=f"eval failed, continuing training: {exc}",
                    )

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
                monitor.log_warning(
                    iteration=iteration,
                    message="local checkpoint save failed.",
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
                        monitor.log_warning(
                            iteration=iteration,
                            message=f"HF upload queued for iteration {iteration}.",
                        )
                    else:
                        hf_checkpointer.upload_checkpoint_files(
                            iteration=iteration,
                            model_path=model_path,
                            buffer_path=buffer_path,
                            metadata_path=metadata_path,
                            keep_last_n=cfg_int("keep_last_n_hf_checkpoints"),
                        )
                        monitor.log_warning(
                            iteration=iteration,
                            message=f"HF checkpoint uploaded for iteration {iteration}.",
                        )
                except (OSError, ValueError):
                    monitor.log_warning(
                        iteration=iteration,
                        message="HF upload failed for this iteration.",
                    )
            if cfg_bool("export_onnx"):
                try:
                    export_onnx(system.model, cfg_str("onnx_path"), device=device)
                except (OSError, RuntimeError, ValueError):
                    monitor.log_warning(
                        iteration=iteration,
                        message="ONNX export failed for this iteration.",
                    )

            cleanup_old_log_versions(
                log_dir=log_dir,
                run_name="ataxx_zero",
                keep_last_n=cfg_int("keep_last_n_log_versions"),
            )
    finally:
        if hf_upload_executor is not None:
            for future in hf_upload_futures:
                try:
                    future.result()
                except Exception:
                    log("A queued HF upload failed.")
            hf_upload_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
