from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.db.models import User
from api.deps.auth import get_admin_user_dep, get_current_user_dep
from api.deps.identity import get_identity_service_dep
from api.modules.identity.schemas import (
    BotProfileListResponse,
    PublicPlayerListResponse,
    UserCreateRequest,
    UserListResponse,
    UserResponse,
)
from api.modules.identity.service import IdentityService

router = APIRouter(prefix="/identity", tags=["identity"])
IDENTITY_SERVICE_DEP = Depends(get_identity_service_dep)
CURRENT_USER_DEP = Depends(get_current_user_dep)
ADMIN_USER_DEP = Depends(get_admin_user_dep)


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create User (Admin)",
    description="Creates a user record. Requires admin privileges.",
    responses={
        201: {"description": "User created successfully."},
        401: {"description": "Missing or invalid access token."},
        403: {"description": "Admin privileges required."},
        409: {"description": "Username or email already exists."},
    },
)
async def post_user(
    request: UserCreateRequest,
    identity_service: IdentityService = IDENTITY_SERVICE_DEP,
    admin_user: User = ADMIN_USER_DEP,
) -> UserResponse:
    del admin_user
    try:
        user = await identity_service.create_user(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return UserResponse.model_validate(user)


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get User",
    description="Returns a user by ID. Allowed for self or admin.",
    responses={
        200: {"description": "User found."},
        401: {"description": "Missing or invalid access token."},
        403: {"description": "Not allowed to view this user."},
        404: {"description": "User not found."},
    },
)
async def get_user(
    user_id: UUID,
    identity_service: IdentityService = IDENTITY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> UserResponse:
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to view this user.",
        )
    user = await identity_service.get_user(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )
    return UserResponse.model_validate(user)


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List Users (Admin)",
    description="Returns user list with optional limit. Requires admin privileges.",
    responses={
        200: {"description": "User list returned."},
        401: {"description": "Missing or invalid access token."},
        403: {"description": "Admin privileges required."},
    },
)
async def list_users(
    limit: int = 50,
    offset: int = 0,
    identity_service: IdentityService = IDENTITY_SERVICE_DEP,
    admin_user: User = ADMIN_USER_DEP,
) -> UserListResponse:
    del admin_user
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    total, users = await identity_service.list_users(limit=safe_limit, offset=safe_offset)
    items = [UserResponse.model_validate(user) for user in users]
    return UserListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


@router.get(
    "/bots",
    response_model=BotProfileListResponse,
    summary="List Playable Bots",
    description="Returns active bot users with enabled bot profile config.",
    responses={
        200: {"description": "Bot list returned."},
        401: {"description": "Missing or invalid access token."},
    },
)
async def list_playable_bots(
    limit: int = 50,
    offset: int = 0,
    identity_service: IdentityService = IDENTITY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> BotProfileListResponse:
    del current_user
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    total, items = await identity_service.list_playable_bots(
        limit=safe_limit,
        offset=safe_offset,
    )
    return BotProfileListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )


@router.get(
    "/players",
    response_model=PublicPlayerListResponse,
    summary="List Public Players",
    description="Returns active public players (humans and visible bots) with optional username filter.",
    responses={
        200: {"description": "Player list returned."},
        401: {"description": "Missing or invalid access token."},
    },
)
async def list_public_players(
    limit: int = 50,
    offset: int = 0,
    q: str | None = None,
    identity_service: IdentityService = IDENTITY_SERVICE_DEP,
    current_user: User = CURRENT_USER_DEP,
) -> PublicPlayerListResponse:
    del current_user
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    total, items = await identity_service.list_public_players(
        limit=safe_limit,
        offset=safe_offset,
        query=q,
    )
    return PublicPlayerListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        has_more=(safe_offset + len(items)) < total,
    )
