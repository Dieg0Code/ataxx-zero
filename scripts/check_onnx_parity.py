from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from model.system import AtaxxZero


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check parity between torch checkpoint and ONNX model.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to torch checkpoint (.ckpt/.pt).")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument("--samples", type=int, default=32, help="Number of random samples.")
    parser.add_argument("--policy-tol", type=float, default=2e-3, help="Policy max-abs tolerance.")
    parser.add_argument("--value-tol", type=float, default=2e-3, help="Value max-abs tolerance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _extract_model_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hparams = checkpoint.get("hparams")
    if not isinstance(hparams, dict):
        return {}
    allowed = {"d_model", "nhead", "num_layers", "dim_feedforward", "dropout"}
    return {key: hparams[key] for key in allowed if key in hparams}


def _load_system(checkpoint_path: Path) -> AtaxxZero:
    from model.system import AtaxxZero

    if checkpoint_path.suffix == ".ckpt":
        return AtaxxZero.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint format: expected dictionary.")
    state_dict_obj = payload.get("state_dict")
    if not isinstance(state_dict_obj, dict):
        raise ValueError("Checkpoint dictionary must contain key 'state_dict'.")
    system = AtaxxZero(**_extract_model_kwargs(payload))
    system.load_state_dict(state_dict_obj)
    return system


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()

    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for parity checks. Install with `uv add --group dev onnxruntime`."
        ) from exc

    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    system = _load_system(checkpoint_path)
    system.eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}

    batch = max(1, int(args.samples))
    board = torch.randn(batch, 4, 7, 7, dtype=torch.float32)
    mask = torch.ones(batch, system.num_actions, dtype=torch.float32)

    with torch.no_grad():
        torch_policy, torch_value = system.model(board, action_mask=mask)
    torch_policy_np = torch_policy.cpu().numpy()
    torch_value_np = torch_value.cpu().numpy()

    ort_inputs: dict[str, Any] = {"board": board.cpu().numpy()}
    if "action_mask" in input_names:
        ort_inputs["action_mask"] = mask.cpu().numpy()
    ort_raw = session.run(None, ort_inputs)
    output_names = [out.name for out in session.get_outputs()]
    ort_out = dict(zip(output_names, ort_raw, strict=True))
    ort_policy_np = np.asarray(ort_out.get("policy", ort_raw[0]), dtype=np.float32)
    ort_value_np = np.asarray(ort_out.get("value", ort_raw[-1]), dtype=np.float32)

    policy_diff = float(np.max(np.abs(torch_policy_np - ort_policy_np)))
    value_diff = float(np.max(np.abs(torch_value_np - ort_value_np)))

    print(f"policy_max_abs_diff={policy_diff:.6f}")
    print(f"value_max_abs_diff={value_diff:.6f}")

    if policy_diff > float(args.policy_tol) or value_diff > float(args.value_tol):
        raise SystemExit(
            f"Parity check failed. policy_diff={policy_diff:.6f} value_diff={value_diff:.6f}"
        )
    print("Parity check passed.")


if __name__ == "__main__":
    main()
