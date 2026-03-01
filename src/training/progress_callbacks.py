from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning import Callback

from training.monitor import TrainingMonitor


class EpochPulseCallback(Callback):
    """Emit sparse epoch-level progress lines during each fit call."""

    def __init__(self, monitor: TrainingMonitor, pulse_every: int = 2) -> None:
        self.monitor = monitor
        self.pulse_every = max(1, int(pulse_every))
        self.iteration = 0

    def set_iteration(self, iteration: int) -> None:
        self.iteration = int(iteration)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = int(trainer.current_epoch) + 1
        max_epochs_raw = trainer.max_epochs
        max_epochs = int(max_epochs_raw) if max_epochs_raw is not None else epoch
        if epoch % self.pulse_every != 0 and epoch != max_epochs:
            return
        self.monitor.log_epoch(
            iteration=self.iteration,
            epoch=epoch,
            max_epochs=max_epochs,
            logged_metrics=trainer.callback_metrics,
        )
