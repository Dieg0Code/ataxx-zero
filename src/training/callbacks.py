from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning import Callback


class OptimizerStateTransfer(Callback):
    """Persist optimizer state across multiple `trainer.fit()` calls."""

    def __init__(self) -> None:
        self._saved_state: dict[str, object] | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._saved_state is None:
            return
        if len(trainer.optimizers) == 0:
            return
        trainer.optimizers[0].load_state_dict(self._saved_state)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if len(trainer.optimizers) == 0:
            return
        self._saved_state = trainer.optimizers[0].state_dict()

