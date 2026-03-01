from __future__ import annotations

from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as functional
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import optim

from game.actions import ACTION_SPACE
from model.transformer import AtaxxTransformerNet


class AtaxxZero(pl.LightningModule):
    """Lightning system for AlphaZero-style policy/value training."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        value_loss_coeff: float = 0.5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        scheduler_type: str = "cosine",
        lr_gamma: float = 0.1,
        milestones: list[int] | None = None,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        if milestones is None:
            milestones = [30, 60, 90]

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.value_loss_coeff = value_loss_coeff
        self.scheduler_type = scheduler_type
        self.lr_gamma = lr_gamma
        self.milestones = milestones
        self.max_epochs = max_epochs

        self.model = AtaxxTransformerNet(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.example_input_array = torch.zeros(1, 4, 7, 7)

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "x" in kwargs:
            x_obj = kwargs["x"]
        elif len(args) > 0:
            x_obj = args[0]
        else:
            raise ValueError("forward expects a board tensor as first argument.")
        if not isinstance(x_obj, torch.Tensor):
            raise TypeError("forward expected torch.Tensor input.")
        action_mask_obj = kwargs.get("action_mask")
        if action_mask_obj is None and len(args) > 1:
            action_mask_obj = args[1]
        if action_mask_obj is not None and not isinstance(action_mask_obj, torch.Tensor):
            raise TypeError("forward expected action_mask as torch.Tensor.")
        return self.model(x_obj, action_mask=action_mask_obj)

    def _resolve_batch_obj(self, args: tuple[object, ...], kwargs: dict[str, object], caller: str) -> object:
        if "batch" in kwargs:
            return kwargs["batch"]
        if len(args) > 0:
            return args[0]
        raise ValueError(f"{caller} expects a batch.")

    def _coerce_train_batch(self, batch_obj: object, caller: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch_obj, Sequence):
            raise TypeError(f"{caller} expected a sequence batch.")
        if len(batch_obj) != 3:
            raise ValueError(f"{caller} expected 3 elements: (boards, target_pis, target_vs).")

        boards_obj = batch_obj[0]
        target_pis_obj = batch_obj[1]
        target_vs_obj = batch_obj[2]
        if not isinstance(boards_obj, torch.Tensor):
            raise TypeError(f"{caller} expected boards as torch.Tensor.")
        if not isinstance(target_pis_obj, torch.Tensor):
            raise TypeError(f"{caller} expected target_pis as torch.Tensor.")
        if not isinstance(target_vs_obj, torch.Tensor):
            raise TypeError(f"{caller} expected target_vs as torch.Tensor.")
        return boards_obj, target_pis_obj, target_vs_obj

    def _common_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        boards, target_pis, target_vs = batch
        legal_mask = (target_pis > 0).to(dtype=boards.dtype)
        has_legal_support = torch.sum(legal_mask, dim=1, keepdim=True) > 0
        if not bool(torch.all(has_legal_support).item()):
            # Defensive fallback: keep logits finite even if a malformed target row is all zeros.
            legal_mask = torch.where(has_legal_support, legal_mask, torch.ones_like(legal_mask))

        pi_logits, v_pred = self(boards, action_mask=legal_mask)

        loss_v = functional.mse_loss(v_pred.view(-1), target_vs.view(-1))
        log_probs = functional.log_softmax(pi_logits, dim=1)
        loss_pi = -torch.sum(target_pis * log_probs) / target_pis.size(0)
        loss = loss_pi + (self.value_loss_coeff * loss_v)

        with torch.no_grad():
            pred_actions = torch.argmax(pi_logits, dim=1)
            target_actions = torch.argmax(target_pis, dim=1)
            policy_accuracy = (pred_actions == target_actions).float().mean()
            value_mae = functional.l1_loss(v_pred.view(-1), target_vs.view(-1))

        return {
            "loss": loss,
            "loss_value": loss_v,
            "loss_policy": loss_pi,
            "policy_accuracy": policy_accuracy,
            "value_mae": value_mae,
        }

    def training_step(
        self,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        batch_obj = self._resolve_batch_obj(args=args, kwargs=kwargs, caller="training_step")
        batch_coerced = self._coerce_train_batch(batch_obj=batch_obj, caller="training_step")
        metrics = self._common_step(batch_coerced)
        if getattr(self, "_trainer", None) is not None:
            self.log_dict(
                {
                    "train/loss": metrics["loss"],
                    "train/loss_value": metrics["loss_value"],
                    "train/loss_policy": metrics["loss_policy"],
                    "train/policy_accuracy": metrics["policy_accuracy"],
                    "train/value_mae": metrics["value_mae"],
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        return metrics["loss"]

    def validation_step(
        self,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        batch_obj = self._resolve_batch_obj(args=args, kwargs=kwargs, caller="validation_step")
        batch_coerced = self._coerce_train_batch(batch_obj=batch_obj, caller="validation_step")
        metrics = self._common_step(batch_coerced)
        if getattr(self, "_trainer", None) is not None:
            self.log_dict(
                {
                    "val/loss": metrics["loss"],
                    "val/loss_value": metrics["loss_value"],
                    "val/loss_policy": metrics["loss_policy"],
                    "val/policy_accuracy": metrics["policy_accuracy"],
                    "val/value_mae": metrics["value_mae"],
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        return metrics["loss"]

    def on_train_epoch_end(self) -> None:
        optimizers = self.trainer.optimizers if self.trainer is not None else []
        if len(optimizers) == 0:
            return
        current_lr = float(optimizers[0].param_groups[0]["lr"])
        self.log("train/lr", current_lr, prog_bar=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler_type = self.scheduler_type.lower()
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "multistep":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=list(self.milestones),
                gamma=self.lr_gamma,
            )
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_obj = self._resolve_batch_obj(args=args, kwargs=kwargs, caller="predict_step")
        if isinstance(batch_obj, Sequence) and len(batch_obj) > 0:
            batch_obj = batch_obj[0]
        if not isinstance(batch_obj, torch.Tensor):
            raise TypeError("predict_step expected torch.Tensor batch.")
        policy_logits, value = self(batch_obj)
        policy = torch.softmax(policy_logits, dim=1)
        return policy, value

    @property
    def num_actions(self) -> int:
        return ACTION_SPACE.num_actions
