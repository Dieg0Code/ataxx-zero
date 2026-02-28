from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from api.db.models import User
from api.deps.auth import get_auth_service_dep, get_current_user_dep
from api.deps.matchmaking import get_matchmaking_service_dep
from api.modules.auth.service import AuthService
from api.modules.matchmaking.schemas import (
    QueueDecisionResponse,
    QueueJoinResponse,
    QueueLeaveResponse,
    QueueStatusResponse,
)
from api.modules.matchmaking.service import MatchmakingService

router = APIRouter(prefix="/matchmaking", tags=["matchmaking"])
MATCHMAKING_SERVICE_DEP = Depends(get_matchmaking_service_dep)
CURRENT_USER_DEP = Depends(get_current_user_dep)
AUTH_SERVICE_DEP = Depends(get_auth_service_dep)


@router.post(
    "/queue/join",
    response_model=QueueJoinResponse,
    summary="Join Ranked Queue",
    description="Queues the authenticated user for ranked matchmaking against humans and bots in MMR range.",
)
async def post_join_queue(
    service: MatchmakingService = MATCHMAKING_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> QueueJoinResponse:
    try:
        return await service.join_ranked_queue(actor_user=current_user)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get(
    "/queue/status",
    response_model=QueueStatusResponse,
    summary="Get Queue Status",
    description="Returns current queue state. If waiting, server attempts a match before responding.",
)
async def get_queue_status(
    service: MatchmakingService = MATCHMAKING_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> QueueStatusResponse:
    return await service.get_status(actor_user=current_user)


@router.post(
    "/queue/leave",
    response_model=QueueLeaveResponse,
    summary="Leave Queue",
    description="Cancels the authenticated user's waiting queue entry.",
)
async def post_leave_queue(
    service: MatchmakingService = MATCHMAKING_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> QueueLeaveResponse:
    return await service.leave_queue(actor_user=current_user)


@router.post(
    "/queue/accept",
    response_model=QueueDecisionResponse,
    summary="Accept Matched Game",
    description="Accepts the currently matched game for authenticated user.",
)
async def post_accept_matched_game(
    service: MatchmakingService = MATCHMAKING_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> QueueDecisionResponse:
    try:
        return await service.accept_current_match(actor_user=current_user)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/queue/reject",
    response_model=QueueDecisionResponse,
    summary="Reject Matched Game",
    description="Rejects currently matched game. Rejector leaves queue; other matched humans return to waiting.",
)
async def post_reject_matched_game(
    service: MatchmakingService = MATCHMAKING_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> QueueDecisionResponse:
    try:
        return await service.reject_current_match(actor_user=current_user)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.websocket("/queue/ws")
async def queue_ws(
    websocket: WebSocket,
    token: str | None = Query(default=None),
    service: MatchmakingService = MATCHMAKING_SERVICE_DEP,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> None:
    if token is None or len(token.strip()) == 0:
        await websocket.close(code=4401, reason="Missing token.")
        return

    try:
        current_user = await auth_service.get_user_from_access_token(token)
    except PermissionError:
        await websocket.close(code=4401, reason="Invalid token.")
        return

    await websocket.accept()
    await websocket.send_json({"type": "queue.subscribed"})

    try:
        while True:
            status_payload = await service.get_status(actor_user=current_user)
            await websocket.send_json(
                {
                    "type": "queue.status",
                    "payload": status_payload.model_dump(mode="json"),
                }
            )
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.5)
            except TimeoutError:
                continue
    except WebSocketDisconnect:
        return
