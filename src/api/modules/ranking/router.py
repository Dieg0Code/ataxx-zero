from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.db.models import User
from api.deps.auth import get_admin_user_dep
from api.deps.ranking import get_ranking_service_dep
from api.modules.ranking.schemas import (
    LeaderboardEntryResponse,
    LeaderboardListResponse,
    RatingResponse,
    SeasonCreateRequest,
    SeasonResponse,
)
from api.modules.ranking.service import RankingService

router = APIRouter(prefix="/ranking", tags=["ranking"])
RANKING_SERVICE_DEP = Depends(get_ranking_service_dep)
ADMIN_USER_DEP = Depends(get_admin_user_dep)


@router.post(
    "/seasons",
    response_model=SeasonResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Season (Admin)",
    description="Creates a ranking season. Requires admin privileges.",
    responses={
        201: {
            "description": "Season created.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "1a2b3c4d-1111-2222-3333-444455556666",
                        "name": "Season 1",
                        "is_active": True,
                        "started_at": "2026-02-22T12:00:00Z",
                        "ended_at": None,
                    }
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Admin privileges required.",
            "content": {"application/json": {"example": {"detail": "Admin privileges required."}}},
        },
    },
)
async def post_season(
    request: SeasonCreateRequest,
    ranking_service: RankingService = RANKING_SERVICE_DEP,
    admin_user: User = ADMIN_USER_DEP,
) -> SeasonResponse:
    del admin_user
    season = await ranking_service.create_season(request)
    return SeasonResponse.model_validate(season)


@router.get(
    "/seasons/active",
    response_model=SeasonResponse,
    summary="Get Active Season",
    description="Returns the currently active season.",
    responses={
        200: {
            "description": "Active season returned.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "1a2b3c4d-1111-2222-3333-444455556666",
                        "name": "Season 1",
                        "is_active": True,
                        "started_at": "2026-02-22T12:00:00Z",
                        "ended_at": None,
                    }
                }
            },
        },
        404: {
            "description": "No active season found.",
            "content": {
                "application/json": {
                    "example": {"detail": "No active season found."},
                }
            },
        },
    },
)
async def get_active_season(
    ranking_service: RankingService = RANKING_SERVICE_DEP,
) -> SeasonResponse:
    season = await ranking_service.get_active_season()
    if season is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active season found.",
        )
    return SeasonResponse.model_validate(season)


@router.get(
    "/ratings/{user_id}/{season_id}",
    response_model=RatingResponse,
    summary="Get User Rating",
    description="Returns (or initializes) user rating for a season.",
    responses={
        200: {
            "description": "Rating returned.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "9f2a6be5-0ef8-4d54-9300-6e6be5f77d50",
                        "user_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                        "season_id": "1a2b3c4d-1111-2222-3333-444455556666",
                        "mmr": 1200.0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "games_played": 0,
                    }
                }
            },
        }
    },
)
async def get_rating(
    user_id: UUID,
    season_id: UUID,
    ranking_service: RankingService = RANKING_SERVICE_DEP,
) -> RatingResponse:
    rating = await ranking_service.get_or_create_rating(user_id=user_id, season_id=season_id)
    return RatingResponse.model_validate(rating)


@router.get(
    "/leaderboard/{season_id}",
    response_model=LeaderboardListResponse,
    summary="Get Leaderboard",
    description="Returns season leaderboard entries.",
    responses={
        200: {
            "description": "Leaderboard returned.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "user_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                            "username": "player_one",
                            "mmr": 1342.7,
                            "wins": 20,
                            "losses": 8,
                            "draws": 2,
                            "games_played": 30,
                            "rank": 1,
                        }
                    ]
                }
            },
        }
    },
)
async def get_leaderboard(
    season_id: UUID,
    limit: int = 100,
    offset: int = 0,
    ranking_service: RankingService = RANKING_SERVICE_DEP,
) -> LeaderboardListResponse:
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    total, entries = await ranking_service.get_leaderboard(
        season_id=season_id,
        limit=safe_limit,
        offset=safe_offset,
    )
    items = [LeaderboardEntryResponse.model_validate(entry) for entry in entries]
    return LeaderboardListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


@router.post(
    "/leaderboard/{season_id}/recompute",
    response_model=list[LeaderboardEntryResponse],
    summary="Recompute Leaderboard (Admin)",
    description="Rebuilds leaderboard from ratings for a season. Requires admin privileges.",
    responses={
        200: {
            "description": "Leaderboard recomputed.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "user_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                            "username": "player_one",
                            "mmr": 1342.7,
                            "wins": 20,
                            "losses": 8,
                            "draws": 2,
                            "games_played": 30,
                            "rank": 1,
                        }
                    ]
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Admin privileges required.",
            "content": {"application/json": {"example": {"detail": "Admin privileges required."}}},
        },
    },
)
async def post_recompute_leaderboard(
    season_id: UUID,
    limit: int = 100,
    ranking_service: RankingService = RANKING_SERVICE_DEP,
    admin_user: User = ADMIN_USER_DEP,
) -> list[LeaderboardEntryResponse]:
    del admin_user
    entries = await ranking_service.recompute_leaderboard(
        season_id=season_id,
        limit=limit,
    )
    return [LeaderboardEntryResponse.model_validate(entry) for entry in entries]
