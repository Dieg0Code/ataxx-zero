from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Ataxx model checkpoint (.ckpt/.pt) to ONNX.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to input checkpoint (.ckpt or .pt).")
    parser.add_argument("--output", required=True, help="Path to output .onnx file.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


class _OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, board: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(board, action_mask=action_mask)


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()

    from game.actions import ACTION_SPACE
    from game.constants import OBSERVATION_CHANNELS
    from model.checkpoint_compat import (
        adapt_state_dict_observation_channels,
        extract_checkpoint_state_dict,
        extract_model_kwargs,
    )
    from model.system import AtaxxZero

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint format: expected dictionary.")
    state_dict_obj = extract_checkpoint_state_dict(payload)
    system = AtaxxZero(**extract_model_kwargs(payload))
    system.load_state_dict(
        adapt_state_dict_observation_channels(
            state_dict_obj,
            target_channels=int(system.model.num_input_channels),
        )
    )

    system.eval()
    wrapper = _OnnxExportWrapper(system.model).eval()

    dummy_board = torch.randn(1, OBSERVATION_CHANNELS, 7, 7, dtype=torch.float32)
    dummy_mask = torch.ones(1, ACTION_SPACE.num_actions, dtype=torch.float32)

    torch.onnx.export(
        model=wrapper,
        args=(dummy_board, dummy_mask),
        f=str(output_path),
        export_params=True,
        opset_version=int(args.opset),
        input_names=["board", "action_mask"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "action_mask": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )
    print(f"ONNX exported: {output_path}")


if __name__ == "__main__":
    main()
