from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, Response

from api.db.enums import GameSource, SampleSplit
from api.deps.training_samples import get_training_samples_service_dep
from api.modules.training_samples.schemas import (
    IngestGameSamplesResponse,
    TrainingSampleCreateRequest,
    TrainingSampleListResponse,
    TrainingSampleResponse,
    TrainingSamplesStatsResponse,
)
from api.modules.training_samples.service import TrainingSamplesService

router = APIRouter(prefix="/training", tags=["training"])
TRAINING_SAMPLES_SERVICE_DEP = Depends(get_training_samples_service_dep)


@router.post(
    "/samples",
    response_model=TrainingSampleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Training Sample",
    description="Creates a single training sample linked to an existing game.",
    responses={
        201: {
            "description": "Training sample created.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "a00e6f62-f89b-41f5-8b5f-59d2075f58a8",
                        "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "move_index": 12,
                        "split": "train",
                        "source": "self_play",
                        "z_value": 1.0,
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        404: {
            "description": "Game not found.",
            "content": {"application/json": {"example": {"detail": "Game not found: <uuid>"}}},
        },
    },
)
async def post_sample(
    request: TrainingSampleCreateRequest,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> TrainingSampleResponse:
    try:
        sample = await service.create_sample(request)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return TrainingSampleResponse.model_validate(sample)


@router.get(
    "/samples",
    response_model=TrainingSampleListResponse,
    summary="List Training Samples",
    description="Lists training samples with optional filters by split and game_id.",
    responses={
        200: {
            "description": "Training sample list returned.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "a00e6f62-f89b-41f5-8b5f-59d2075f58a8",
                            "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                            "move_index": 12,
                            "split": "train",
                            "source": "self_play",
                            "z_value": 1.0,
                            "created_at": "2026-02-22T12:00:00Z",
                        }
                    ]
                }
            },
        }
    },
)
async def list_samples(
    limit: int = 100,
    offset: int = 0,
    split: SampleSplit | None = None,
    game_id: UUID | None = None,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> TrainingSampleListResponse:
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    total, samples = await service.list_samples(
        limit=safe_limit,
        offset=safe_offset,
        split=split,
        game_id=game_id,
    )
    items = [TrainingSampleResponse.model_validate(sample) for sample in samples]
    return TrainingSampleListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


@router.get(
    "/samples/stats",
    response_model=TrainingSamplesStatsResponse,
    summary="Get Training Samples Stats",
    description="Returns aggregate counts by split and source.",
    responses={
        200: {
            "description": "Training sample stats returned.",
            "content": {
                "application/json": {
                    "example": {
                        "total": 10000,
                        "by_split": {"train": 8000, "val": 1500, "test": 500},
                        "by_source": {"self_play": 9400, "human": 600},
                    }
                }
            },
        }
    },
)
async def get_samples_stats(
    split: SampleSplit | None = None,
    game_id: UUID | None = None,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> TrainingSamplesStatsResponse:
    total, by_split, by_source = await service.stats(split=split, game_id=game_id)
    return TrainingSamplesStatsResponse(
        total=total,
        by_split=by_split,
        by_source=by_source,
    )


@router.get(
    "/samples/export.ndjson",
    response_class=PlainTextResponse,
    summary="Export Training Samples NDJSON",
    description="Exports training samples in NDJSON format.",
    responses={
        200: {
            "description": "NDJSON export returned.",
            "content": {
                "text/plain": {
                    "example": (
                        "{\"game_id\":\"8bcbf808-c8ab-4f75-95e8-f5f0871500af\","
                        "\"move_index\":12,\"split\":\"train\",\"source\":\"self_play\"}\n"
                    )
                }
            },
        }
    },
)
async def export_samples_ndjson(
    split: SampleSplit | None = None,
    game_id: UUID | None = None,
    limit: int = 5000,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> str:
    return await service.export_ndjson(split=split, game_id=game_id, limit=limit)


@router.get(
    "/samples/export.npz",
    summary="Export Training Samples NPZ",
    description="Exports training samples in compressed NPZ format.",
    responses={200: {"description": "NPZ export returned as binary payload."}},
)
async def export_samples_npz(
    split: SampleSplit | None = None,
    game_id: UUID | None = None,
    limit: int = 5000,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> Response:
    payload = await service.export_npz(split=split, game_id=game_id, limit=limit)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="training_samples.npz"',
        },
    )


@router.get(
    "/samples/{sample_id}",
    response_model=TrainingSampleResponse,
    summary="Get Training Sample",
    description="Returns a training sample by ID.",
    responses={
        200: {
            "description": "Training sample returned.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "a00e6f62-f89b-41f5-8b5f-59d2075f58a8",
                        "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "move_index": 12,
                        "split": "train",
                        "source": "self_play",
                        "z_value": 1.0,
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        404: {
            "description": "Training sample not found.",
            "content": {
                "application/json": {
                    "example": {"detail": "Training sample not found: <uuid>"},
                }
            },
        },
    },
)
async def get_sample(
    sample_id: UUID,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> TrainingSampleResponse:
    sample = await service.get_sample(sample_id)
    if sample is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training sample not found: {sample_id}",
        )
    return TrainingSampleResponse.model_validate(sample)


@router.post(
    "/samples/ingest-game/{game_id}",
    response_model=IngestGameSamplesResponse,
    summary="Ingest Samples From Game",
    description="Converts stored game moves into training samples for a chosen split/source.",
    responses={
        200: {
            "description": "Samples ingested from game.",
            "content": {
                "application/json": {
                    "example": {
                        "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "created_count": 64,
                        "split": "train",
                        "source": "self_play",
                    }
                }
            },
        },
        400: {
            "description": "Game is not ingestible (not finished/no moves/etc).",
            "content": {
                "application/json": {
                    "example": {"detail": "Game must be finished before ingestion."},
                }
            },
        },
        404: {
            "description": "Game not found.",
            "content": {"application/json": {"example": {"detail": "Game not found: <uuid>"}}},
        },
    },
)
async def post_ingest_game(
    game_id: UUID,
    split: SampleSplit = SampleSplit.TRAIN,
    source: GameSource = GameSource.SELF_PLAY,
    overwrite: bool = False,
    service: TrainingSamplesService = TRAINING_SAMPLES_SERVICE_DEP,
) -> IngestGameSamplesResponse:
    try:
        samples = await service.ingest_from_game(
            game_id=game_id,
            split=split,
            source=source,
            overwrite=overwrite,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return IngestGameSamplesResponse(
        game_id=game_id,
        created_count=len(samples),
        split=split,
        source=source,
    )
