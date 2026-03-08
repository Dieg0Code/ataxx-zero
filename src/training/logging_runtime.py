from __future__ import annotations

from pathlib import Path

from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger

from training.config_runtime import log


def build_training_logger(log_dir: Path) -> Logger:
    """
    Prefer TensorBoard when available, but do not block training on missing extras.
    """
    try:
        return TensorBoardLogger(save_dir=str(log_dir), name="ataxx_zero")
    except ModuleNotFoundError as exc:
        log(f"TensorBoard logger unavailable ({exc}); falling back to CSVLogger.")
        return CSVLogger(save_dir=str(log_dir), name="ataxx_zero")


__all__ = ["build_training_logger"]
