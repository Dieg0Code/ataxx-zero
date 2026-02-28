from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.db.models import User
from api.deps.auth import get_current_user_dep
from api.deps.inference import get_inference_service_dep
from api.deps.matches import get_matches_service_dep
from api.modules.gameplay.schemas import GameResponse
from api.modules.matches.schemas import (
    MatchAdvanceBotResponse,
    MatchCreateRequest,
    MatchMovePayload,
    MatchMoveRequest,
    MatchMoveResponse,
    MatchStateResponse,
)
from api.modules.matches.service import MatchesService
from game.serialization import board_to_state

router = APIRouter(prefix="/matches", tags=["matches"])
MATCHES_SERVICE_DEP = Depends(get_matches_service_dep)
CURRENT_USER_DEP = Depends(get_current_user_dep)


@router.post(
    "",
    response_model=GameResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Match",
    description="Creates an authoritative match. player1 is bound to authenticated user.",
    responses={
        201: {
            "description": "Match created.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "player1_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                        "player2_id": "8f12a4a6-2f9e-40fd-9f97-bcf8f5e6aace",
                        "status": "in_progress",
                        "winner_id": None,
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Invalid player ownership constraints.",
            "content": {"application/json": {"example": {"detail": "player1 must be the authenticated user."}}},
        },
    },
)
async def post_match(
    request: MatchCreateRequest,
    service: MatchesService = MATCHES_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> GameResponse:
    try:
        game = await service.create_match(request, actor_user_id=current_user.id)
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
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    return GameResponse.model_validate(game)


@router.get(
    "/{game_id}",
    response_model=GameResponse,
    summary="Get Match",
    description="Returns match metadata for participants or admin.",
    responses={
        200: {
            "description": "Match found.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "player1_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                        "player2_id": "8f12a4a6-2f9e-40fd-9f97-bcf8f5e6aace",
                        "status": "in_progress",
                        "winner_id": None,
                        "created_at": "2026-02-22T12:00:00Z",
                    }
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Not allowed to view this match.",
            "content": {"application/json": {"example": {"detail": "You are not allowed to view this match."}}},
        },
        404: {
            "description": "Match not found.",
            "content": {"application/json": {"example": {"detail": "Match not found: <uuid>"}}},
        },
    },
)
async def get_match(
    game_id: UUID,
    service: MatchesService = MATCHES_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> GameResponse:
    try:
        await service.ensure_can_view_match(game_id=game_id, actor_user=current_user)
        game = await service.get_match(game_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    if game is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match not found: {game_id}",
        )
    return GameResponse.model_validate(game)


@router.get(
    "/{game_id}/state",
    response_model=MatchStateResponse,
    summary="Get Match State",
    description="Returns current board, next side and legal moves for participants or admin.",
    responses={
        200: {
            "description": "Match state returned.",
            "content": {
                "application/json": {
                    "example": {
                        "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "status": "in_progress",
                        "board": {
                            "grid": [[1, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0]],
                            "current_player": 1,
                            "turn_count": 0,
                        },
                        "next_player_side": "p1",
                        "legal_moves": [{"r1": 0, "c1": 0, "r2": 1, "c2": 1}],
                    }
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Not allowed to view this match.",
            "content": {"application/json": {"example": {"detail": "You are not allowed to view this match."}}},
        },
        404: {
            "description": "Match not found.",
            "content": {"application/json": {"example": {"detail": "Match not found: <uuid>"}}},
        },
    },
)
async def get_match_state(
    game_id: UUID,
    service: MatchesService = MATCHES_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> MatchStateResponse:
    try:
        await service.ensure_can_view_match(game_id=game_id, actor_user=current_user)
        game = await service.get_match(game_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    if game is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match not found: {game_id}",
        )
    board = await service.get_current_board(game_id=game_id)
    next_side = service._to_side(board.current_player)
    legal_moves = [
        MatchMovePayload(r1=r1, c1=c1, r2=r2, c2=c2)
        for r1, c1, r2, c2 in board.get_valid_moves()
    ]
    return MatchStateResponse(
        game_id=game_id,
        status=game.status,
        board=board_to_state(board),
        next_player_side=next_side,
        legal_moves=legal_moves,
    )


@router.post(
    "/{game_id}/moves",
    response_model=MatchMoveResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Match Move",
    description="Applies one authoritative move from authenticated player.",
    responses={
        201: {
            "description": "Move accepted and persisted.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "b43d6731-0545-4d8f-b48f-9707bf7a9516",
                        "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                        "move_index": 17,
                        "move": {"r1": 3, "c1": 3, "r2": 4, "c2": 4},
                        "player_side": "p1",
                        "created_at": "2026-02-22T12:30:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Illegal move payload.",
            "content": {"application/json": {"example": {"detail": "Illegal move for current board state."}}},
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Not participant or invalid turn.",
            "content": {"application/json": {"example": {"detail": "It is not your turn."}}},
        },
        404: {
            "description": "Match not found.",
            "content": {"application/json": {"example": {"detail": "Match not found: <uuid>"}}},
        },
    },
)
async def post_match_move(
    game_id: UUID,
    request: MatchMoveRequest,
    service: MatchesService = MATCHES_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> MatchMoveResponse:
    try:
        move = await service.submit_move(
            game_id=game_id,
            request=request,
            actor_user_id=current_user.id,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return MatchMoveResponse.model_validate(move)


@router.post(
    "/{game_id}/advance-bot",
    response_model=MatchAdvanceBotResponse,
    summary="Advance Bot Turn",
    description="Applies one server-side move for the bot if it is currently the bot turn.",
    responses={
        200: {"description": "Bot turn processed."},
        400: {"description": "Invalid bot state or profile."},
        401: {"description": "Missing or invalid access token."},
        403: {"description": "Not allowed to access this match."},
        404: {"description": "Match not found."},
        503: {"description": "Inference service unavailable for model bot."},
    },
)
async def post_advance_bot(
    game_id: UUID,
    service: MatchesService = MATCHES_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> MatchAdvanceBotResponse:
    inference_service = None
    try:
        inference_service = get_inference_service_dep()
    except HTTPException:
        inference_service = None
    try:
        move = await service.advance_bot_turn(
            game_id=game_id,
            actor_user=current_user,
            inference_service=inference_service,
        )
        game = await service.get_match(game_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if game is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match not found: {game_id}",
        )
    if move is None:
        return MatchAdvanceBotResponse(
            game_id=game_id,
            applied=False,
            status=game.status,
            message="No bot move applied.",
            move=None,
        )
    return MatchAdvanceBotResponse(
        game_id=game_id,
        applied=True,
        status=game.status,
        message="Bot move applied.",
        move=MatchMoveResponse.model_validate(move),
    )
