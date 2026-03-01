from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

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


class _OnnxIoLike(Protocol):
    name: str


class _OnnxSessionLike(Protocol):
    def get_inputs(self) -> list[_OnnxIoLike]:
        ...

    def get_outputs(self) -> list[_OnnxIoLike]:
        ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        ...


class InferenceService:
    """Checkpoint-backed inference service for Ataxx move selection."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        onnx_path: str | Path | None = None,
        prefer_onnx: bool = True,
        device: str = "auto",
        mcts_sims: int = 160,
        c_puct: float = 1.5,
        model_kwargs: ModelInitKwargs | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.prefer_onnx = bool(prefer_onnx)
        if not self.checkpoint_path.exists() and (
            self.onnx_path is None or not self.onnx_path.exists()
        ):
            raise FileNotFoundError(
                f"No inference artifacts found. checkpoint={self.checkpoint_path} onnx={self.onnx_path}"
            )

        self.device = self._resolve_device(device)
        self.mcts_sims = max(1, int(mcts_sims))
        self.c_puct = float(c_puct)
        self.model_kwargs: ModelInitKwargs = model_kwargs or {}

        self.system: AtaxxZero | None = None
        if self.checkpoint_path.exists():
            self.system = self._load_system()
            self.system.eval()
            self.system.to(self.device)

        self._onnx_session: _OnnxSessionLike | None = None
        self._onnx_last_error: str | None = None
        self._onnx_input_names: set[str] = set()
        if self.prefer_onnx and self.onnx_path is not None and self.onnx_path.exists():
            self._init_onnx_session()

        if self.system is None and self._onnx_session is None:
            raise ValueError(
                "Inference initialization failed: neither torch checkpoint nor ONNX session is available."
            )
        self._mcts: MCTS | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_system(self) -> AtaxxZero:
        ckpt = self.checkpoint_path
        if ckpt.suffix == ".ckpt":
            try:
                return AtaxxZero.load_from_checkpoint(str(ckpt), map_location=self.device)
            except RuntimeError as exc:
                raise ValueError(
                    "Checkpoint incompatible con architecture policy_head espacial; "
                    "reentrena o usa carga parcial manual (strict=False)."
                ) from exc

        checkpoint = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid .pt checkpoint format: expected dictionary.")
        state_dict_obj = checkpoint.get("state_dict")
        if not isinstance(state_dict_obj, dict):
            raise ValueError("Checkpoint dictionary must contain key 'state_dict'.")

        system = AtaxxZero(**self.model_kwargs)
        try:
            system.load_state_dict(state_dict_obj)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint incompatible con architecture policy_head espacial; "
                "reentrena o usa carga parcial manual (strict=False)."
            ) from exc
        return system

    def _init_onnx_session(self) -> None:
        if self.onnx_path is None:
            return
        session = self._load_onnx_session(self.onnx_path)
        self._onnx_session = session
        self._onnx_input_names = {inp.name for inp in session.get_inputs()}

    def _load_onnx_session(self, onnx_path: Path) -> _OnnxSessionLike:
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ValueError(
                "onnxruntime is required to run ONNX inference. Install it with `uv add --group api onnxruntime`."
            ) from exc

        available = set(ort.get_available_providers())
        providers: list[str] = []
        if self.device == "cuda" and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return ort.InferenceSession(str(onnx_path), providers=providers)

    def _ensure_mcts(self) -> MCTS:
        if self.system is None:
            raise ValueError("Strong mode requires a torch checkpoint.")
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

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        exp_logits = np.exp(shifted)
        denom = float(np.sum(exp_logits))
        if denom <= 0.0 or not np.isfinite(denom):
            return np.zeros_like(logits, dtype=np.float32)
        return (exp_logits / denom).astype(np.float32)

    def _fast_result_onnx(self, board: AtaxxBoard) -> InferenceResult:
        if self._onnx_session is None:
            raise ValueError("ONNX session is not initialized.")
        mask_np = self._legal_action_mask(board).astype(np.float32)
        obs_np = board.get_observation().astype(np.float32, copy=False)[None, ...]
        inputs: dict[str, Any] = {"board": obs_np}
        if "action_mask" in self._onnx_input_names:
            inputs["action_mask"] = mask_np[None, ...]

        raw_outputs = self._onnx_session.run(None, inputs)
        output_names = [out.name for out in self._onnx_session.get_outputs()]
        outputs = dict(zip(output_names, raw_outputs, strict=True))

        policy_logits: np.ndarray
        value_scalar: float
        if "policy" in outputs and "value" in outputs:
            policy_logits = np.asarray(outputs["policy"], dtype=np.float32).reshape(-1)
            value_scalar = float(np.asarray(outputs["value"], dtype=np.float32).reshape(-1)[0])
        else:
            arrays = [np.asarray(item, dtype=np.float32) for item in raw_outputs]
            policy_candidates = [arr for arr in arrays if arr.size == ACTION_SPACE.num_actions]
            scalar_candidates = [arr for arr in arrays if arr.size == 1]
            if len(policy_candidates) == 0 or len(scalar_candidates) == 0:
                raise ValueError("Unexpected ONNX output format for policy/value.")
            policy_logits = policy_candidates[0].reshape(-1)
            value_scalar = float(scalar_candidates[0].reshape(-1)[0])

        if "action_mask" not in self._onnx_input_names:
            policy_logits = np.where(mask_np > 0.0, policy_logits, -1e9).astype(np.float32)
        policy = self._stable_softmax(policy_logits)

        if not np.all(np.isfinite(policy)) or float(np.sum(policy)) <= 0.0:
            legal_indices = np.flatnonzero(mask_np > 0)
            action_idx = int(legal_indices[0]) if legal_indices.size > 0 else ACTION_SPACE.pass_index
        else:
            action_idx = int(np.argmax(policy))
            if mask_np[action_idx] <= 0:
                legal_indices = np.flatnonzero(mask_np > 0)
                action_idx = int(legal_indices[0]) if legal_indices.size > 0 else ACTION_SPACE.pass_index

        move = ACTION_SPACE.decode(action_idx)
        return InferenceResult(move=move, action_idx=action_idx, value=value_scalar, mode="fast")

    def _fast_result(self, board: AtaxxBoard) -> InferenceResult:
        if self._onnx_session is not None:
            try:
                return self._fast_result_onnx(board)
            except Exception as exc:
                # Fallback to torch checkpoint path if ONNX fails at runtime.
                self._onnx_last_error = str(exc)

        if self.system is None:
            raise ValueError("Fast inference unavailable: no torch checkpoint and ONNX failed.")
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
        if self.system is None:
            # If no torch model is available, degrade gracefully to fast ONNX/Torch.
            return self._fast_result(board)
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
