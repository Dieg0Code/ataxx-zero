from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

import numpy as np

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.types import Move

if TYPE_CHECKING:
    import torch.nn as nn

    from engine.mcts import MCTS

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


class _SystemLike(Protocol):
    @property
    def model(self) -> nn.Module:
        ...

    def eval(self) -> _SystemLike:
        ...

    def to(self, device: str) -> _SystemLike:
        ...

    def load_state_dict(self, state_dict: dict[str, object]) -> object:
        ...


class _CheckpointSystemAdapter:
    """Minimal runtime wrapper to use plain torch modules as inference systems."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    @property
    def model(self) -> nn.Module:
        return self._model

    def eval(self) -> _CheckpointSystemAdapter:
        self._model.eval()
        return self

    def to(self, device: str) -> _CheckpointSystemAdapter:
        self._model.to(device)
        return self

    def load_state_dict(self, state_dict: dict[str, object]) -> object:
        return self._model.load_state_dict(state_dict)


@lru_cache(maxsize=1)
def _get_torch_module() -> ModuleType | None:
    """Import torch lazily so API startup does not hard-fail in lightweight runtimes."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


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

        self.system: _SystemLike | None = None
        self._model_input_channels = 4
        if self.checkpoint_path.exists():
            self.system = self._load_system()
            self.system.eval()
            self.system.to(self.device)
            self._model_input_channels = int(
                getattr(self.system.model, "num_input_channels", 4)
            )

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
            torch_module = _get_torch_module()
            if torch_module is not None and bool(torch_module.cuda.is_available()):
                return "cuda"
            return "cpu"
        return device

    @staticmethod
    def _require_torch() -> ModuleType:
        torch_module = _get_torch_module()
        if torch_module is None:
            raise ValueError(
                "Torch is required for checkpoint-backed inference. "
                "Use ONNX artifacts or install torch in this runtime."
            )
        return torch_module

    @staticmethod
    def _is_legacy_state_dict(state_dict: dict[str, Any]) -> bool:
        has_legacy_policy = "model.policy_head.1.weight" in state_dict
        has_spatial_policy = "model.policy_src_proj.weight" in state_dict
        input_weight = state_dict.get("model.input_proj.weight")
        input_channels = None
        if hasattr(input_weight, "shape"):
            shape = tuple(input_weight.shape)
            if len(shape) == 2:
                input_channels = int(shape[1])
        return has_legacy_policy and not has_spatial_policy and input_channels == 3

    @staticmethod
    def _extract_arch_kwargs(raw_kwargs: ModelInitKwargs) -> dict[str, Any]:
        allowed = ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout")
        return {key: raw_kwargs[key] for key in allowed if key in raw_kwargs}

    @staticmethod
    def _extract_model_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        # Training checkpoints prefix model params with `model.` (Lightning module layout).
        # Runtime inference uses the raw network, so we strip this prefix when present.
        if all(key.startswith("model.") for key in state_dict):
            return {key.removeprefix("model."): value for key, value in state_dict.items()}
        return state_dict

    def _build_legacy_system(self) -> _SystemLike:
        from inference.legacy_model import LegacyAtaxxSystem

        return LegacyAtaxxSystem(**self._extract_arch_kwargs(self.model_kwargs))

    def _build_spatial_system(self) -> _SystemLike:
        from model.transformer import AtaxxTransformerNet

        model = AtaxxTransformerNet(**self._extract_arch_kwargs(self.model_kwargs))
        return _CheckpointSystemAdapter(model)

    def _load_system(self) -> _SystemLike:
        torch_module = self._require_torch()
        ckpt = self.checkpoint_path
        if ckpt.suffix == ".ckpt":
            from model.system import AtaxxZero

            try:
                return AtaxxZero.load_from_checkpoint(str(ckpt), map_location=self.device)
            except RuntimeError as exc:
                raise ValueError(
                    "Checkpoint incompatible con architecture policy_head espacial; "
                    "reentrena o usa carga parcial manual (strict=False)."
                ) from exc

        checkpoint = torch_module.load(str(ckpt), map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid .pt checkpoint format: expected dictionary.")
        state_dict_obj = checkpoint.get("state_dict")
        if not isinstance(state_dict_obj, dict):
            raise ValueError("Checkpoint dictionary must contain key 'state_dict'.")

        system = self._build_spatial_system()
        try:
            system.load_state_dict(self._extract_model_state_dict(state_dict_obj))
        except RuntimeError as exc:
            if self._is_legacy_state_dict(state_dict_obj):
                legacy_system = self._build_legacy_system()
                try:
                    legacy_system.load_state_dict(state_dict_obj)
                    return legacy_system
                except RuntimeError as legacy_exc:
                    raise ValueError(
                        "Checkpoint incompatible con architecture policy_head espacial; "
                        "reentrena o usa carga parcial manual (strict=False)."
                    ) from legacy_exc
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
        from engine.mcts import MCTS

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
        torch_module = self._require_torch()
        mask_np = self._legal_action_mask(board)
        obs = board.get_observation()
        if obs.shape[0] != self._model_input_channels:
            obs = obs[: self._model_input_channels]

        obs_tensor = torch_module.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch_module.from_numpy(mask_np).unsqueeze(0).to(self.device)
        with torch_module.no_grad():
            policy_logits, value_tensor = self.system.model(obs_tensor, action_mask=mask_tensor)

        policy = (
            torch_module.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
        )
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
        if self._model_input_channels != 4:
            # Legacy checkpoints were trained with 3-channel observations and do
            # not support the current MCTS path that batches 4-channel states.
            return self._fast_result(board)
        torch_module = self._require_torch()
        mcts = self._ensure_mcts()
        probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
        action_idx = int(np.argmax(probs))
        move = ACTION_SPACE.decode(action_idx)

        # Value still comes from raw net (current-player perspective), which is stable and cheap.
        mask_np = self._legal_action_mask(board)
        obs = board.get_observation()
        if obs.shape[0] != self._model_input_channels:
            obs = obs[: self._model_input_channels]
        obs_tensor = torch_module.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch_module.from_numpy(mask_np).unsqueeze(0).to(self.device)
        with torch_module.no_grad():
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
