from __future__ import annotations

from functools import lru_cache

from api.db.models import ModelVersion
from api.inference_artifacts import resolve_artifact_uri
from inference.service import InferenceService


def _runtime_config_from_base(base_service: InferenceService | None) -> tuple[str, int, float, bool]:
    if base_service is None:
        return "auto", 160, 1.5, True
    return (
        base_service.device,
        base_service.mcts_sims,
        base_service.c_puct,
        base_service.prefer_onnx,
    )


@lru_cache(maxsize=32)
def _build_cached_inference_service(
    checkpoint_path: str,
    onnx_path: str,
    device: str,
    mcts_sims: int,
    c_puct: float,
    prefer_onnx: bool,
) -> InferenceService:
    return InferenceService(
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path if onnx_path else None,
        device=device,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
        prefer_onnx=prefer_onnx,
    )


def resolve_model_inference_service(
    *,
    version: ModelVersion,
    base_service: InferenceService | None,
) -> InferenceService:
    checkpoint_path = resolve_artifact_uri(
        version.checkpoint_uri,
        default_repo_id=version.hf_repo_id,
        default_revision=version.hf_revision,
    )
    onnx_path = resolve_artifact_uri(
        version.onnx_uri,
        default_repo_id=version.hf_repo_id,
        default_revision=version.hf_revision,
    )
    if checkpoint_path is None and onnx_path is None:
        raise ValueError(
            f"Model version '{version.name}' has no usable checkpoint_uri/onnx_uri configured."
        )

    # Keep runtime knobs aligned with the API default inference service so
    # per-account model selection does not silently change move quality.
    device, mcts_sims, c_puct, prefer_onnx = _runtime_config_from_base(base_service)
    return _build_cached_inference_service(
        checkpoint_path=checkpoint_path or "",
        onnx_path=onnx_path or "",
        device=device,
        mcts_sims=mcts_sims,
        c_puct=c_puct,
        prefer_onnx=prefer_onnx,
    )


def prewarm_model_inference_service(
    *,
    version: ModelVersion,
    base_service: InferenceService | None,
) -> InferenceService:
    service = resolve_model_inference_service(
        version=version,
        base_service=base_service,
    )
    service.warmup(mode="fast")
    return service


__all__ = ["prewarm_model_inference_service", "resolve_model_inference_service"]
