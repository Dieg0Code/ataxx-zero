from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from api.config import Settings, get_settings
from inference.service import InferenceService


@lru_cache(maxsize=1)
def _build_inference_service(
    checkpoint_path: str,
    onnx_path: str,
    device: str,
    mcts_sims: int,
    c_puct: float,
    prefer_onnx: bool,
) -> InferenceService:
    return InferenceService(
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path if onnx_path.strip() else None,
        device=device,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
        prefer_onnx=prefer_onnx,
    )


def get_inference_service_dep() -> InferenceService:
    settings: Settings = get_settings()
    try:
        return _build_inference_service(
            checkpoint_path=settings.model_checkpoint_path,
            onnx_path=settings.model_onnx_path,
            device=settings.inference_device,
            mcts_sims=settings.inference_mcts_sims,
            c_puct=settings.inference_c_puct,
            prefer_onnx=settings.inference_prefer_onnx,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {exc}",
        ) from exc
