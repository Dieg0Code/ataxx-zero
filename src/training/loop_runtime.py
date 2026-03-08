from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader

from training.bootstrap import generate_imitation_data
from training.callbacks import OptimizerStateTransfer
from training.config_runtime import (
    TrainerPrecision,
    cfg_bool,
    cfg_float,
    cfg_int,
    cfg_str,
    log,
)
from training.monitor import TrainingMonitor
from training.progress_callbacks import EpochPulseCallback
from training.trainer_runtime import (
    build_trainer,
    is_ddp_rendezvous_timeout,
    resolve_trainer_precision,
)
from training.warmup_runtime import allocate_warmup_games, resolve_warmup_levels

if TYPE_CHECKING:
    from data.replay_buffer import ReplayBuffer, TrainingExample
    from model.system import AtaxxZero


def resolve_eval_levels() -> list[str]:
    csv_levels = cfg_str("eval_heuristic_levels").strip()
    if csv_levels == "":
        return [cfg_str("eval_heuristic_level")]
    levels: list[str] = []
    for part in csv_levels.split(","):
        level = part.strip()
        if level != "" and level not in levels:
            levels.append(level)
    return levels if len(levels) > 0 else [cfg_str("eval_heuristic_level")]


def restore_system_from_checkpoint(system: AtaxxZero, checkpoint_path: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint payload: expected dictionary.")
    state_dict_obj = payload.get("state_dict")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("Checkpoint payload missing state_dict dictionary.")
    system.load_state_dict(state_dict_obj)


def prepare_train_val_examples(
    *,
    buffer: ReplayBuffer,
    split_seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    from data.dataset import split_train_val_examples
    from data.replay_buffer import sample_recent_mix

    all_examples = buffer.get_all()
    train_examples, val_examples = split_train_val_examples(
        all_examples=all_examples,
        val_split=cfg_float("val_split"),
        shuffle=cfg_bool("shuffle_train_val_split"),
        seed=split_seed,
    )
    train_examples = sample_recent_mix(
        train_examples,
        recent_fraction=cfg_float("train_recent_fraction"),
        recent_window_fraction=cfg_float("train_recent_window_fraction"),
        seed=split_seed + 17,
        sample_size=len(train_examples),
    )
    return train_examples, val_examples


def build_train_loader(
    train_examples: list[TrainingExample],
    *,
    device: str,
) -> DataLoader[object]:
    from data.dataset import AtaxxDataset

    dataset = AtaxxDataset(
        buffer=None,
        augment=True,
        reference_buffer=False,
        examples=train_examples,
    )
    if cfg_int("num_workers") > 0:
        return DataLoader(
            dataset,
            batch_size=cfg_int("batch_size"),
            shuffle=True,
            num_workers=cfg_int("num_workers"),
            persistent_workers=cfg_bool("persistent_workers"),
            pin_memory=(device == "cuda"),
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=cfg_int("batch_size"),
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=(device == "cuda"),
    )


def build_val_loader(
    val_examples: list[TrainingExample],
    *,
    device: str,
) -> DataLoader[object] | None:
    from data.dataset import ValidationDataset

    val_dataset = ValidationDataset(
        buffer=None,
        split=cfg_float("val_split"),
        examples=val_examples,
    )
    if len(val_dataset) == 0:
        return None
    if cfg_int("num_workers") > 0:
        return DataLoader(
            val_dataset,
            batch_size=cfg_int("batch_size"),
            shuffle=False,
            num_workers=cfg_int("num_workers"),
            persistent_workers=cfg_bool("persistent_workers"),
            pin_memory=(device == "cuda"),
            prefetch_factor=2,
        )
    return DataLoader(
        val_dataset,
        batch_size=cfg_int("batch_size"),
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=(device == "cuda"),
    )


def fit_with_ddp_fallback(
    *,
    system: AtaxxZero,
    train_loader: DataLoader[object],
    val_loader: DataLoader[object] | None,
    epochs: int,
    trainer_accelerator: str,
    trainer_devices: int,
    trainer_strategy: str,
    trainer_precision: TrainerPrecision,
    checkpoint_callback: ModelCheckpoint,
    lr_monitor: LearningRateMonitor,
    logger: Logger,
    optimizer_transfer: OptimizerStateTransfer,
    epoch_pulse: EpochPulseCallback,
) -> tuple[pl.Trainer, str, int, str, TrainerPrecision]:
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
        # Hosted notebooks are prone to DDP rendezvous flakiness; downgrade
        # rather than losing the whole training window.
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


def run_warmup_if_needed(
    *,
    start_iteration: int,
    system: AtaxxZero,
    buffer: ReplayBuffer,
    trainer_accelerator: str,
    trainer_devices: int,
    trainer_strategy: str,
    trainer_precision: TrainerPrecision,
    checkpoint_callback: ModelCheckpoint,
    lr_monitor: LearningRateMonitor,
    logger: Logger,
    device: str,
    optimizer_transfer: OptimizerStateTransfer,
    monitor: TrainingMonitor,
    epoch_pulse: EpochPulseCallback,
) -> tuple[str, int, str, TrainerPrecision]:
    warmup_games = cfg_int("warmup_games")
    warmup_epochs = cfg_int("warmup_epochs")
    if start_iteration != 0 or warmup_games <= 0 or warmup_epochs <= 0:
        return trainer_accelerator, trainer_devices, trainer_strategy, trainer_precision

    warmup_rng = torch.Generator().manual_seed(cfg_int("seed"))
    rng_seed = int(torch.randint(0, 2**31, (1,), generator=warmup_rng).item())
    warmup_plan = allocate_warmup_games(
        total_games=warmup_games,
        levels=resolve_warmup_levels(
            raw_levels_csv=cfg_str("warmup_heuristic_levels"),
            fallback_level=cfg_str("warmup_heuristic_level"),
        ),
    )
    warmup_examples: list[TrainingExample] = []
    for level_idx, (warmup_level, warmup_level_games) in enumerate(warmup_plan):
        if warmup_level_games <= 0:
            continue
        warmup_examples.extend(
            generate_imitation_data(
                n_games=warmup_level_games,
                seed=rng_seed + (level_idx * 1_009),
                heuristic_level=cast(Any, warmup_level),
            ),
        )
    warmup_levels_log = " ".join(
        f"{level}:{games}" for level, games in warmup_plan if games > 0
    )
    if warmup_levels_log != "":
        log(f"Warmup level mix -> {warmup_levels_log}")
    buffer.save_game(warmup_examples)
    monitor.log_warmup(examples=len(warmup_examples), games=warmup_games)
    train_examples, val_examples = prepare_train_val_examples(
        buffer=buffer,
        split_seed=cfg_int("seed"),
    )
    train_loader = build_train_loader(train_examples, device=device)
    val_loader = build_val_loader(val_examples, device=device)
    (
        _warmup_trainer,
        trainer_accelerator,
        trainer_devices,
        trainer_strategy,
        trainer_precision,
    ) = fit_with_ddp_fallback(
        system=system,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=warmup_epochs,
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
    return trainer_accelerator, trainer_devices, trainer_strategy, trainer_precision


__all__ = [
    "build_train_loader",
    "build_val_loader",
    "fit_with_ddp_fallback",
    "prepare_train_val_examples",
    "resolve_eval_levels",
    "restore_system_from_checkpoint",
    "run_warmup_if_needed",
]
