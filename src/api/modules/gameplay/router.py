from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.exc import IntegrityError

from agents.heuristic import heuristic_move
from agents.random_agent import random_move
from api.db.enums import GameStatus
from api.db.models import Game, User
from api.deps.auth import get_auth_service_dep, get_current_user_dep
from api.deps.gameplay import get_gameplay_service_dep
from api.deps.inference import get_inference_service_dep
from api.modules.auth.service import AuthService
from api.modules.gameplay.schemas import (
    GameCreateRequest,
    GameListResponse,
    GameReplayResponse,
    GameResponse,
    ManualMoveRequest,
    MovePayload,
    MoveRequest,
    MoveResponse,
    StoredMoveResponse,
)
from api.modules.gameplay.service import GameplayService
from api.modules.gameplay.ws import gameplay_ws_hub
from game.actions import ACTION_SPACE
from game.serialization import board_from_state
from inference.service import InferenceService

router = APIRouter(prefix="/gameplay", tags=["gameplay"])
INFERENCE_SERVICE_DEP = Depends(get_inference_service_dep)
GAMEPLAY_SERVICE_DEP = Depends(get_gameplay_service_dep)
CURRENT_USER_DEP = Depends(get_current_user_dep)
AUTH_SERVICE_DEP = Depends(get_auth_service_dep)
logger = logging.getLogger(__name__)


async def _to_game_response(
    gameplay_service: GameplayService,
    game: Game,
) -> GameResponse:
    get_usernames = getattr(gameplay_service, "get_player_usernames", None)
    if get_usernames is None:
        player1_username, player2_username = None, None
    else:
        player1_username, player2_username = await get_usernames(game)
    validated = GameResponse.model_validate(game)
    validated.player1_username = player1_username
    validated.player2_username = player2_username
    return validated


async def _broadcast_move_applied(
    gameplay_service: GameplayService,
    game_id: UUID,
    stored_move: StoredMoveResponse,
) -> None:
    game = await gameplay_service.get_game(game_id)
    if game is None:
        return
    game_payload = await _to_game_response(gameplay_service, game)
    await gameplay_ws_hub.broadcast(
        game_id=game_id,
        payload={
            "type": "game.move.applied",
            "game_id": str(game_id),
            "move": stored_move.model_dump(mode="json"),
            "game": game_payload.model_dump(mode="json"),
        },
    )


@router.post(
    "/move",
    response_model=MoveResponse,
    summary="Predict Move",
    description="Runs inference for a provided board payload. Does not persist game state.",
    responses={
        200: {
            "description": "Prediction computed.",
            "content": {
                "application/json": {
                    "example": {
                        "move": {"r1": 3, "c1": 3, "r2": 4, "c2": 4},
                        "action_idx": 128,
                        "value": 0.42,
                        "mode": "mcts",
                    }
                }
            },
        },
        400: {
            "description": "Invalid board payload.",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid board payload: invalid current_player"},
                }
            },
        },
    },
)
def post_move(
    request: MoveRequest,
    inference_service: InferenceService = INFERENCE_SERVICE_DEP,
) -> MoveResponse:
    try:
        board = board_from_state(request.board.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid board payload: {exc}",
        ) from exc

    mode = request.mode
    if mode in {"fast", "strong"}:
        result = inference_service.predict(board=board, mode=mode)
        move_payload: MovePayload | None = None
        if result.move is not None:
            r1, c1, r2, c2 = result.move
            move_payload = MovePayload(r1=r1, c1=c1, r2=r2, c2=c2)

        return MoveResponse(
            move=move_payload,
            action_idx=result.action_idx,
            value=result.value,
            mode=result.mode,
        )

    rng = np.random.default_rng()
    if mode == "random":
        move = random_move(board=board, rng=rng)
    else:
        level = mode.removeprefix("heuristic_")
        move = heuristic_move(board=board, rng=rng, level=level)

    if move is None:
        return MoveResponse(
            move=None,
            action_idx=ACTION_SPACE.pass_index,
            value=0.0,
            mode=mode,
        )
    r1, c1, r2, c2 = move
    return MoveResponse(
        move=MovePayload(r1=r1, c1=c1, r2=r2, c2=c2),
        action_idx=ACTION_SPACE.encode(move),
        value=0.0,
        mode=mode,
    )


@router.post(
    "/games",
    response_model=GameResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Game",
    description="Creates a game record. For non-admin users, player1_id is forced to the authenticated user.",
    responses={
        201: {
            "description": "Game created.",
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
            "description": "Invalid ownership constraints for player1_id.",
            "content": {"application/json": {"example": {"detail": "player1 must be the authenticated user."}}},
        },
    },
)
async def post_game(
    request: GameCreateRequest,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> GameResponse:
    try:
        game = await gameplay_service.create_game(request, actor_user=current_user)
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
    return await _to_game_response(gameplay_service, game)


@router.get(
    "/games/{game_id}",
    response_model=GameResponse,
    summary="Get Game",
    description="Returns a game by ID. Allowed for participants or admin.",
    responses={
        200: {
            "description": "Game found.",
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
            "description": "Not allowed to view this game.",
            "content": {"application/json": {"example": {"detail": "You are not allowed to view this game."}}},
        },
        404: {
            "description": "Game not found.",
            "content": {"application/json": {"example": {"detail": "Game not found: <uuid>"}}},
        },
    },
)
async def get_game(
    game_id: UUID,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> GameResponse:
    try:
        game = await gameplay_service.ensure_can_view_game(
            game_id=game_id,
            actor_user=current_user,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Game row is invalid/corrupt for API model: {exc}",
        ) from exc
    return await _to_game_response(gameplay_service, game)


@router.get(
    "/games",
    response_model=GameListResponse,
    summary="List Games",
    description="Lists games visible to authenticated user. Admin sees all games.",
    responses={
        200: {
            "description": "Game list returned.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                            "player1_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                            "player2_id": "8f12a4a6-2f9e-40fd-9f97-bcf8f5e6aace",
                            "status": "in_progress",
                            "winner_id": None,
                            "created_at": "2026-02-22T12:00:00Z",
                        }
                    ]
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
    },
)
async def list_games(
    limit: int = 20,
    offset: int = 0,
    status_filter: Annotated[list[str] | None, Query(alias="status")] = None,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> GameListResponse:
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    statuses = None
    if status_filter:
        allowed_statuses = {enum_value.value: enum_value for enum_value in GameStatus}
        invalid_values = [value for value in status_filter if value not in allowed_statuses]
        if invalid_values:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid status filter values: {', '.join(invalid_values)}",
            )
        statuses = [allowed_statuses[value] for value in status_filter]
    try:
        total, games = await gameplay_service.list_games(
            limit=safe_limit,
            offset=safe_offset,
            actor_user=current_user,
            statuses=statuses,
        )
    except ValueError:
        logger.warning(
            "Returning empty game list due to invalid/corrupt rows.",
            extra={"user_id": str(current_user.id), "limit": safe_limit, "offset": safe_offset},
        )
        return GameListResponse(
            items=[],
            total=0,
            limit=safe_limit,
            offset=safe_offset,
            has_more=False,
        )
    items: list[GameResponse] = []
    for game in games:
        try:
            items.append(await _to_game_response(gameplay_service, game))
        except ValueError:
            # Defensive path for legacy/corrupt rows. Do not fail the whole list.
            logger.warning("Skipping invalid game row during list.", extra={"game_id": str(game.id)})
    return GameListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


@router.post(
    "/games/{game_id}/move",
    response_model=StoredMoveResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Apply Inference Move",
    description="Computes and stores one inference move for a game. Allowed for participants or admin.",
    responses={
        201: {
            "description": "Move stored.",
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
            "description": "Invalid board payload.",
            "content": {"application/json": {"example": {"detail": "Invalid board payload: malformed grid"}}},
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Not allowed to mutate this game.",
            "content": {"application/json": {"example": {"detail": "You are not allowed to mutate this game."}}},
        },
        404: {
            "description": "Game not found.",
            "content": {"application/json": {"example": {"detail": "Game not found: <uuid>"}}},
        },
    },
)
async def post_game_move(
    game_id: UUID,
    request: MoveRequest,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    inference_service: InferenceService = INFERENCE_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> StoredMoveResponse:
    if request.mode not in {"fast", "strong"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Persisted inference move only supports mode 'fast' or 'strong'.",
        )

    try:
        board = board_from_state(request.board.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid board payload: {exc}",
        ) from exc

    inference = inference_service.predict(board=board, mode=request.mode)
    try:
        stored = await gameplay_service.record_inference_move(
            game_id=game_id,
            board=board,
            inference=inference,
            actor_user=current_user,
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

    payload = StoredMoveResponse.model_validate(stored)
    await _broadcast_move_applied(
        gameplay_service=gameplay_service,
        game_id=game_id,
        stored_move=payload,
    )
    return payload


@router.post(
    "/games/{game_id}/move/manual",
    response_model=StoredMoveResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Apply Manual Move",
    description="Stores one explicit move payload for a game. Allowed for participants or admin.",
)
async def post_game_manual_move(
    game_id: UUID,
    request: ManualMoveRequest,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> StoredMoveResponse:
    try:
        board = board_from_state(request.board.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid board payload: {exc}",
        ) from exc

    move = (request.move.r1, request.move.c1, request.move.r2, request.move.c2)
    try:
        stored = await gameplay_service.record_manual_move(
            game_id=game_id,
            board=board,
            move=move,
            actor_user=current_user,
            mode=request.mode,
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

    payload = StoredMoveResponse.model_validate(stored)
    await _broadcast_move_applied(
        gameplay_service=gameplay_service,
        game_id=game_id,
        stored_move=payload,
    )
    return payload


@router.websocket("/games/{game_id}/ws")
async def gameplay_game_ws(
    websocket: WebSocket,
    game_id: UUID,
    token: str | None = Query(default=None),
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> None:
    if token is None or len(token.strip()) == 0:
        await websocket.close(code=4401, reason="Missing token.")
        return

    try:
        actor_user = await auth_service.get_user_from_access_token(token)
    except PermissionError:
        await websocket.close(code=4401, reason="Invalid token.")
        return

    try:
        game = await gameplay_service.ensure_can_view_game(
            game_id=game_id,
            actor_user=actor_user,
        )
    except LookupError:
        await websocket.close(code=4404, reason="Game not found.")
        return
    except PermissionError:
        await websocket.close(code=4403, reason="Not allowed.")
        return
    except ValueError:
        await websocket.close(code=4400, reason="Invalid game row.")
        return

    await gameplay_ws_hub.connect(game_id=game_id, websocket=websocket)
    try:
        await gameplay_ws_hub.send_personal(
            websocket,
            {
                "type": "game.subscribed",
                "game_id": str(game_id),
                "status": game.status.value,
            },
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await gameplay_ws_hub.disconnect(game_id=game_id, websocket=websocket)


@router.get(
    "/games/{game_id}/moves",
    response_model=list[StoredMoveResponse],
    summary="List Game Moves",
    description="Returns stored moves for a game. Allowed for participants or admin.",
    responses={
        200: {
            "description": "Move list returned.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "b43d6731-0545-4d8f-b48f-9707bf7a9516",
                            "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                            "move_index": 17,
                            "move": {"r1": 3, "c1": 3, "r2": 4, "c2": 4},
                            "player_side": "p1",
                            "created_at": "2026-02-22T12:30:00Z",
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
            "description": "Not allowed to view this game.",
            "content": {"application/json": {"example": {"detail": "You are not allowed to view this game."}}},
        },
        404: {
            "description": "Game not found.",
            "content": {"application/json": {"example": {"detail": "Game not found: <uuid>"}}},
        },
    },
)
async def list_game_moves(
    game_id: UUID,
    limit: int = 200,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> list[StoredMoveResponse]:
    safe_limit = max(1, min(limit, 500))
    try:
        await gameplay_service.ensure_can_view_game(game_id=game_id, actor_user=current_user)
        moves = await gameplay_service.list_game_moves(game_id=game_id, limit=safe_limit)
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
    return [StoredMoveResponse.model_validate(move) for move in moves]


@router.get(
    "/games/{game_id}/replay",
    response_model=GameReplayResponse,
    summary="Get Game Replay",
    description="Returns game metadata and moves as replay payload. Allowed for participants or admin.",
    responses={
        200: {
            "description": "Replay returned.",
            "content": {
                "application/json": {
                    "example": {
                        "game": {
                            "id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                            "player1_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                            "player2_id": "8f12a4a6-2f9e-40fd-9f97-bcf8f5e6aace",
                            "status": "finished",
                            "winner_id": "fcb1ce84-229e-4ea6-9f31-89cf8a5f58af",
                            "created_at": "2026-02-22T12:00:00Z",
                        },
                        "moves": [
                            {
                                "id": "b43d6731-0545-4d8f-b48f-9707bf7a9516",
                                "game_id": "8bcbf808-c8ab-4f75-95e8-f5f0871500af",
                                "move_index": 17,
                                "move": {"r1": 3, "c1": 3, "r2": 4, "c2": 4},
                                "player_side": "p1",
                                "created_at": "2026-02-22T12:30:00Z",
                            }
                        ],
                    }
                }
            },
        },
        401: {
            "description": "Missing or invalid access token.",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Not allowed to view this game.",
            "content": {"application/json": {"example": {"detail": "You are not allowed to view this game."}}},
        },
        404: {
            "description": "Game not found.",
            "content": {"application/json": {"example": {"detail": "Game not found: <uuid>"}}},
        },
    },
)
async def get_game_replay(
    game_id: UUID,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> GameReplayResponse:
    try:
        game = await gameplay_service.ensure_can_view_game(
            game_id=game_id,
            actor_user=current_user,
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Game row is invalid/corrupt for replay: {exc}",
        ) from exc

    try:
        moves = await gameplay_service.list_game_moves(game_id=game_id, limit=500)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Move rows are invalid/corrupt for replay: {exc}",
        ) from exc

    safe_moves: list[StoredMoveResponse] = []
    for move in moves:
        try:
            safe_moves.append(StoredMoveResponse.model_validate(move))
        except ValueError:
            # Defensive path for legacy/corrupt rows. Keep replay available.
            continue

    try:
        game_payload = await _to_game_response(gameplay_service, game)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Game row is invalid/corrupt for replay payload: {exc}",
        ) from exc

    return GameReplayResponse(
        game=game_payload,
        moves=safe_moves,
    )


@router.delete(
    "/games/{game_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Game",
    description="Deletes a game and its moves. Allowed for participants or admin.",
    responses={
        204: {"description": "Game deleted."},
        401: {"description": "Missing or invalid access token."},
        403: {"description": "Not allowed to delete this game."},
        404: {"description": "Game not found."},
    },
)
async def delete_game(
    game_id: UUID,
    gameplay_service: GameplayService = GAMEPLAY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> None:
    try:
        await gameplay_service.delete_game(game_id=game_id, actor_user=current_user)
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Game row is invalid/corrupt for delete: {exc}",
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Game cannot be deleted due to related records: {exc.orig}",
        ) from exc
