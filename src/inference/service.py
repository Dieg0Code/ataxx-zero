from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch

from engine.mcts import MCTS
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.types import Move
from model.system import AtaxxZero

InferenceMode = Literal["fast", "strong"]


class ModelInitKwargs(TypedDict, total=False):
    learning_rate: float
    weight_decay: float
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    scheduler_type: str
    lr_gamma: float
    milestones: list[int]
    max_epochs: int


@dataclass(frozen=True)
class InferenceResult:
    move: Move | None
    action_idx: int
    value: float
    mode: InferenceMode


class InferenceService:
    """Checkpoint-backed inference service for Ataxx move selection."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        mcts_sims: int = 160,
        c_puct: float = 1.5,
        model_kwargs: ModelInitKwargs | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = self._resolve_device(device)
        self.mcts_sims = max(1, int(mcts_sims))
        self.c_puct = float(c_puct)
        self.model_kwargs: ModelInitKwargs = model_kwargs or {}

        self.system = self._load_system()
        self.system.eval()
        self.system.to(self.device)
        self._mcts: MCTS | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_system(self) -> AtaxxZero:
        ckpt = self.checkpoint_path
        if ckpt.suffix == ".ckpt":
            return AtaxxZero.load_from_checkpoint(str(ckpt), map_location=self.device)

        checkpoint = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid .pt checkpoint format: expected dictionary.")
        state_dict_obj = checkpoint.get("state_dict")
        if not isinstance(state_dict_obj, dict):
            raise ValueError("Checkpoint dictionary must contain key 'state_dict'.")

        system = AtaxxZero(**self.model_kwargs)
        system.load_state_dict(state_dict_obj)
        return system

    def _ensure_mcts(self) -> MCTS:
        if self._mcts is None:
            self._mcts = MCTS(
                model=self.system.model,
                c_puct=self.c_puct,
                n_simulations=self.mcts_sims,
                device=self.device,
            )
        return self._mcts

    @staticmethod
    def _legal_action_mask(board: AtaxxBoard) -> np.ndarray:
        valid_moves = board.get_valid_moves()
        include_pass = len(valid_moves) == 0
        return ACTION_SPACE.mask_from_moves(valid_moves, include_pass=include_pass)

    def _fast_result(self, board: AtaxxBoard) -> InferenceResult:
        mask_np = self._legal_action_mask(board)
        obs = board.get_observation()

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value_tensor = self.system.model(obs_tensor, action_mask=mask_tensor)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
        if not np.all(np.isfinite(policy)):
            legal_indices = np.flatnonzero(mask_np > 0)
            if legal_indices.size == 0:
                action_idx = ACTION_SPACE.pass_index
            else:
                action_idx = int(legal_indices[0])
        else:
            action_idx = int(np.argmax(policy))
            if mask_np[action_idx] <= 0:
                legal_indices = np.flatnonzero(mask_np > 0)
                if legal_indices.size == 0:
                    action_idx = ACTION_SPACE.pass_index
                else:
                    action_idx = int(legal_indices[0])

        move = ACTION_SPACE.decode(action_idx)
        value = float(value_tensor.item())
        return InferenceResult(move=move, action_idx=action_idx, value=value, mode="fast")

    def _strong_result(self, board: AtaxxBoard) -> InferenceResult:
        mcts = self._ensure_mcts()
        probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
        action_idx = int(np.argmax(probs))
        move = ACTION_SPACE.decode(action_idx)

        # Value still comes from raw net (current-player perspective), which is stable and cheap.
        mask_np = self._legal_action_mask(board)
        obs = board.get_observation()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value_tensor = self.system.model(obs_tensor, action_mask=mask_tensor)
        value = float(value_tensor.item())
        return InferenceResult(move=move, action_idx=action_idx, value=value, mode="strong")

    def predict(self, board: AtaxxBoard, *, mode: InferenceMode = "fast") -> InferenceResult:
        if board.is_game_over():
            return InferenceResult(
                move=None,
                action_idx=ACTION_SPACE.pass_index,
                value=0.0,
                mode=mode,
            )
        if mode == "strong":
            return self._strong_result(board)
        if mode == "fast":
            return self._fast_result(board)
        raise ValueError(f"Unsupported inference mode: {mode}")
