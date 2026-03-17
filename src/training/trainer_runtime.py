from __future__ import annotations

from datetime import timedelta

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import DDPStrategy

from game.constants import OBSERVATION_CHANNELS
from training.config_runtime import (
    TrainerPrecision,
    cfg_bool,
    cfg_int,
    log,
    resolve_trainer_precision,
)


def resolve_trainer_hw() -> tuple[str, int, str]:
    from training.config_runtime import cfg_int, cfg_str

    requested_devices = cfg_int("trainer_devices")
    strategy = cfg_str("trainer_strategy")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if int(capability[0]) < 7:
            log(
                "Detected CUDA device with compute capability "
                f"sm_{int(capability[0])}{int(capability[1])}, unsupported by current torch build. "
                "Falling back to CPU.",
            )
            return "cpu", 1, "auto"
        available = max(1, torch.cuda.device_count())
        devices = min(requested_devices, available)
        if requested_devices > available:
            log(
                f"Requested {requested_devices} GPU(s), but only {available} available. Using {devices}.",
            )
        # Avoid distributed rendezvous on single-GPU environments (common in hosted notebooks)
        # when CLI flags request ddp/ddp_spawn unconditionally.
        if devices <= 1 and strategy in {"ddp", "ddp_spawn"}:
            log(f"Requested strategy '{strategy}' requires >1 GPU. Falling back to 'auto'.")
            strategy = "auto"
        elif strategy == "auto" and devices > 1:
            strategy = "ddp"
        return "gpu", devices, strategy
    if requested_devices > 1:
        log("CUDA not available, forcing devices=1 on CPU.")
    if strategy in {"ddp", "ddp_spawn"}:
        log(f"Requested strategy '{strategy}' on CPU. Falling back to 'auto'.")
        strategy = "auto"
    return "cpu", 1, strategy


def is_ddp_rendezvous_timeout(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        ("DistStoreError" in msg or "init_process_group" in msg)
        and ("clients joined" in msg or "Timed out" in msg)
    )


def resolve_trainer_strategy(strategy: str) -> str | DDPStrategy:
    timeout = timedelta(seconds=max(30, cfg_int("ddp_timeout_seconds")))
    if strategy == "ddp":
        return DDPStrategy(timeout=timeout, start_method="popen")
    if strategy == "ddp_spawn":
        return DDPStrategy(timeout=timeout, start_method="spawn")
    return strategy


def build_trainer(
    *,
    epochs: int,
    accelerator: str,
    devices: int,
    strategy: str,
    precision: TrainerPrecision,
    benchmark: bool,
    checkpoint_callback: ModelCheckpoint,
    lr_monitor: LearningRateMonitor,
    logger: Logger,
    extra_callbacks: list[Callback] | None = None,
) -> pl.Trainer:
    callbacks: list[Callback] = [checkpoint_callback, lr_monitor]
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)
    resolved_strategy = resolve_trainer_strategy(strategy)
    return pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=resolved_strategy,
        precision=precision,
        benchmark=benchmark,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=cfg_bool("show_progress_bar"),
        log_every_n_steps=cfg_int("trainer_log_every_n_steps"),
        gradient_clip_val=1.0,
    )


def export_onnx(model: torch.nn.Module, path: str, device: str) -> None:
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, OBSERVATION_CHANNELS, 7, 7, device=device)
    try:
        torch.onnx.export(
            model=model,
            args=(dummy_input,),
            f=path,
            export_params=True,
            opset_version=11,
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes={
                "board": {0: "batch_size"},
                "policy": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )
        log(f"Exported ONNX to {path}")
    except ModuleNotFoundError as exc:
        log(
            "ONNX export skipped: missing dependency "
            f"({exc}). Install with: `uv add onnx onnxscript`.",
        )


__all__ = [
    "build_trainer",
    "export_onnx",
    "is_ddp_rendezvous_timeout",
    "resolve_trainer_hw",
    "resolve_trainer_precision",
    "resolve_trainer_strategy",
]
