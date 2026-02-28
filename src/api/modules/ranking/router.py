from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.db.models import User
from api.deps.auth import get_admin_user_dep
from api.deps.ranking import get_ranking_service_dep
from api.modules.ranking.schemas import (
    LeaderboardEntryResponse,
    LeaderboardListResponse,
    RatingEventListResponse,
    RatingEventResponse,
    RatingResponse,
    SeasonCreateRequest,
    SeasonResponse,
)
from api.modules.ranking.service import RankingService

router = APIRouter(prefix="/ranking", tags=["ranking"])
RANKING_SERVICE_DEP = Depends(get_ranking_service_dep)
ADMIN_USER_DEP = Depends(get_admin_user_dep)

_LEAGUE_INDEX = {"Protocol": 0, "Kernel": 1, "Root": 2}
_DIVISION_INDEX = {"III": 0, "II": 1, "I": 2}


def _ladder_points(league: str | None, division: str | None, lp: int | None) -> int | None:
    if league is None or division is None or lp is None:
        return None
    league_idx = _LEAGUE_INDEX.get(league)
    division_idx = _DIVISION_INDEX.get(division)
    if league_idx is None or division_idx is None:
        return None
    return (league_idx * 3 + division_idx) * 100 + lp


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
    league, division, lp = ranking_service.build_league_snapshot(rating.rating)
    next_major_promo = ranking_service.next_major_promo_name(league, division)
    return RatingResponse(
        id=rating.id,
        user_id=rating.user_id,
        season_id=rating.season_id,
        rating=rating.rating,
        games_played=rating.games_played,
        wins=rating.wins,
        losses=rating.losses,
        draws=rating.draws,
        updated_at=rating.updated_at,
        league=league,
        division=division,
        lp=lp,
        next_major_promo=next_major_promo,
    )


@router.get(
    "/ratings/{user_id}/{season_id}/events",
    response_model=RatingEventListResponse,
    summary="Get Rating Event History",
    description="Returns rating transition events for a player in a season.",
)
async def get_rating_events(
    user_id: UUID,
    season_id: UUID,
    limit: int = 50,
    offset: int = 0,
    ranking_service: RankingService = RANKING_SERVICE_DEP,
) -> RatingEventListResponse:
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    total, events = await ranking_service.get_rating_events(
        user_id=user_id,
        season_id=season_id,
        limit=safe_limit,
        offset=safe_offset,
    )
    items = [RatingEventResponse.model_validate(event) for event in events]
    return RatingEventListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


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
    competitor_filter: str = "all",
    q: str | None = None,
    ranking_service: RankingService = RANKING_SERVICE_DEP,
) -> LeaderboardListResponse:
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    normalized_filter = competitor_filter.lower()
    if normalized_filter not in {"all", "humans", "bots"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="competitor_filter must be one of: all, humans, bots.",
        )
    username_query = q.strip() if q else None
    if username_query == "":
        username_query = None
    total, entries = await ranking_service.get_leaderboard(
        season_id=season_id,
        limit=safe_limit,
        offset=safe_offset,
        competitor_filter=normalized_filter,
        username_query=username_query,
    )
    latest_events = await ranking_service.get_latest_rating_events_for_user_ids(
        season_id=season_id,
        user_ids=[entry.user_id for entry in entries],
    )
    user_info_by_id = await ranking_service.get_user_public_info_for_user_ids(
        [entry.user_id for entry in entries]
    )
    items = []
    for entry in entries:
        user_info = user_info_by_id.get(entry.user_id)
        latest_event = latest_events.get(entry.user_id)
        username = user_info[0] if user_info is not None else None
        is_bot = user_info[1] if user_info is not None else False
        bot_kind = user_info[2] if user_info is not None else None
        recent_lp_delta: int | None = None
        recent_transition_type: str | None = None
        if latest_event is not None:
            before_points = _ladder_points(
                latest_event.before_league,
                latest_event.before_division,
                latest_event.before_lp,
            )
            after_points = _ladder_points(
                latest_event.after_league,
                latest_event.after_division,
                latest_event.after_lp,
            )
            if before_points is not None and after_points is not None:
                recent_lp_delta = after_points - before_points
            recent_transition_type = latest_event.transition_type
        league, division, lp = ranking_service.build_league_snapshot(entry.rating)
        next_major_promo = ranking_service.next_major_promo_name(league, division)
        prestige_title = (
            ranking_service.TOP_PRESTIGE_TITLE if entry.rank == 1 else None
        )
        items.append(
            LeaderboardEntryResponse(
                season_id=entry.season_id,
                user_id=entry.user_id,
                username=username,
                is_bot=is_bot,
                bot_kind=bot_kind,
                rank=entry.rank,
                rating=entry.rating,
                wins=entry.wins,
                losses=entry.losses,
                draws=entry.draws,
                win_rate=entry.win_rate,
                computed_at=entry.computed_at,
                league=league,
                division=division,
                lp=lp,
                recent_lp_delta=recent_lp_delta,
                recent_transition_type=recent_transition_type,
                next_major_promo=next_major_promo,
                prestige_title=prestige_title,
            )
        )
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
    latest_events = await ranking_service.get_latest_rating_events_for_user_ids(
        season_id=season_id,
        user_ids=[entry.user_id for entry in entries],
    )
    user_info_by_id = await ranking_service.get_user_public_info_for_user_ids(
        [entry.user_id for entry in entries]
    )
    response: list[LeaderboardEntryResponse] = []
    for entry in entries:
        user_info = user_info_by_id.get(entry.user_id)
        latest_event = latest_events.get(entry.user_id)
        username = user_info[0] if user_info is not None else None
        is_bot = user_info[1] if user_info is not None else False
        bot_kind = user_info[2] if user_info is not None else None
        recent_lp_delta: int | None = None
        recent_transition_type: str | None = None
        if latest_event is not None:
            before_points = _ladder_points(
                latest_event.before_league,
                latest_event.before_division,
                latest_event.before_lp,
            )
            after_points = _ladder_points(
                latest_event.after_league,
                latest_event.after_division,
                latest_event.after_lp,
            )
            if before_points is not None and after_points is not None:
                recent_lp_delta = after_points - before_points
            recent_transition_type = latest_event.transition_type
        league, division, lp = ranking_service.build_league_snapshot(entry.rating)
        next_major_promo = ranking_service.next_major_promo_name(league, division)
        prestige_title = (
            ranking_service.TOP_PRESTIGE_TITLE if entry.rank == 1 else None
        )
        response.append(
            LeaderboardEntryResponse(
                season_id=entry.season_id,
                user_id=entry.user_id,
                username=username,
                is_bot=is_bot,
                bot_kind=bot_kind,
                rank=entry.rank,
                rating=entry.rating,
                wins=entry.wins,
                losses=entry.losses,
                draws=entry.draws,
                win_rate=entry.win_rate,
                computed_at=entry.computed_at,
                league=league,
                division=division,
                lp=lp,
                recent_lp_delta=recent_lp_delta,
                recent_transition_type=recent_transition_type,
                next_major_promo=next_major_promo,
                prestige_title=prestige_title,
            )
        )
    return response
