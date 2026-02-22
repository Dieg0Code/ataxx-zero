from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from api.config.settings import Settings, get_settings
from api.db.models import User
from api.db.session import get_session
from api.modules.auth.repository import AuthRepository
from api.modules.auth.service import AuthService

SESSION_DEP = Depends(get_session)
SETTINGS_DEP = Depends(get_settings)
bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="bearerAuth",
    description="JWT access token. Format: Bearer <token>",
)
CREDENTIALS_DEP = Depends(bearer_scheme)


def get_auth_service_dep(
    session: AsyncSession = SESSION_DEP,
    settings: Settings = SETTINGS_DEP,
) -> AuthService:
    return AuthService(repository=AuthRepository(session=session), settings=settings)


AUTH_SERVICE_DEP = Depends(get_auth_service_dep)


async def get_current_user_dep(
    credentials: HTTPAuthorizationCredentials | None = CREDENTIALS_DEP,
    auth_service: AuthService = AUTH_SERVICE_DEP,
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token.",
        )
    try:
        user = await auth_service.get_user_from_access_token(credentials.credentials)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    return user


CURRENT_USER_DEP = Depends(get_current_user_dep)


def get_admin_user_dep(current_user: User = CURRENT_USER_DEP) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return current_user
