from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from api.db.models import User
from api.deps.auth import get_auth_service_dep, get_current_user_dep
from api.modules.auth.schemas import (
    AuthLoginRequest,
    AuthLogoutRequest,
    AuthRefreshRequest,
    AuthRegisterRequest,
    AuthTokenPairResponse,
    AuthUserResponse,
)
from api.modules.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
AUTH_SERVICE_DEP = Depends(get_auth_service_dep)
CURRENT_USER_DEP = Depends(get_current_user_dep)


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _refresh_principal(refresh_token: str) -> str:
    return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:24]


@router.post(
    "/register",
    response_model=AuthUserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register User",
    description="Creates a new account with username, optional email and password.",
    responses={
        201: {"description": "User registered successfully."},
        409: {"description": "Username or email already exists."},
    },
)
async def post_register(
    request: AuthRegisterRequest,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> AuthUserResponse:
    try:
        user = await auth_service.register(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return AuthUserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=AuthTokenPairResponse,
    summary="Login",
    description="Authenticates user credentials and returns access + refresh JWT tokens.",
    responses={
        200: {"description": "Authentication successful."},
        401: {"description": "Invalid credentials or inactive user."},
        429: {"description": "Too many login attempts. Rate limit exceeded."},
    },
)
async def post_login(
    http_request: Request,
    request: AuthLoginRequest,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> AuthTokenPairResponse:
    limiter = getattr(http_request.app.state, "auth_rate_limiter", None)
    if limiter is not None:
        decision = limiter.check_login(
            client_ip=_client_ip(http_request),
            principal=request.username_or_email,
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please retry later.",
                headers={"Retry-After": str(decision.retry_after_s)},
            )
    try:
        return await auth_service.login(request)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


@router.post(
    "/refresh",
    response_model=AuthTokenPairResponse,
    summary="Refresh Tokens",
    description="Rotates refresh token and returns a new access/refresh pair.",
    responses={
        200: {"description": "Token refresh successful."},
        401: {"description": "Invalid, revoked, or expired refresh token."},
        429: {"description": "Too many refresh attempts. Rate limit exceeded."},
    },
)
async def post_refresh(
    http_request: Request,
    request: AuthRefreshRequest,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> AuthTokenPairResponse:
    limiter = getattr(http_request.app.state, "auth_rate_limiter", None)
    if limiter is not None:
        decision = limiter.check_refresh(
            client_ip=_client_ip(http_request),
            principal=_refresh_principal(request.refresh_token),
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many refresh attempts. Please retry later.",
                headers={"Retry-After": str(decision.retry_after_s)},
            )
    try:
        return await auth_service.refresh(request.refresh_token)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout",
    description="Revokes a refresh token. Access token remains valid until expiry.",
    responses={204: {"description": "Logout successful."}},
)
async def post_logout(
    request: AuthLogoutRequest,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> Response:
    await auth_service.logout(request.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/me",
    response_model=AuthUserResponse,
    summary="Get Current User",
    description="Returns profile data for the authenticated user.",
    responses={
        200: {"description": "Current user profile."},
        401: {"description": "Missing or invalid access token."},
    },
)
async def get_me(current_user: User = CURRENT_USER_DEP) -> AuthUserResponse:
    return AuthUserResponse.model_validate(current_user)
