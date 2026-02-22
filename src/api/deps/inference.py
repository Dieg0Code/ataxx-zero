from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from api.config import Settings, get_settings
from inference.service import InferenceService


@lru_cache(maxsize=1)
def _build_inference_service(
    checkpoint_path: str,
    device: str,
    mcts_sims: int,
    c_puct: float,
) -> InferenceService:
    return InferenceService(
        checkpoint_path=checkpoint_path,
        device=device,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
    )


def get_inference_service_dep() -> InferenceService:
    settings: Settings = get_settings()
    try:
        return _build_inference_service(
            checkpoint_path=settings.model_checkpoint_path,
            device=settings.inference_device,
            mcts_sims=settings.inference_mcts_sims,
            c_puct=settings.inference_c_puct,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {exc}",
        ) from exc
