from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from api.config.settings import Settings
from api.db.models import AuthRefreshToken, User
from api.modules.auth.repository import AuthRepository
from api.modules.auth.schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthTokenPairResponse,
)
from api.modules.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)


class AuthService:
    def __init__(self, repository: AuthRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self.jwt_kind_access = "access"
        self.jwt_kind_refresh = "refresh"
        self.bearer_type = "bearer"

    async def register(self, payload: AuthRegisterRequest) -> User:
        now = datetime.now(timezone.utc)
        user = User(
            username=payload.username,
            email=payload.email,
            password_hash=hash_password(payload.password),
            is_active=True,
            is_admin=False,
            created_at=now,
            updated_at=now,
        )
        try:
            return await self.repository.create_user(user)
        except IntegrityError as exc:
            raise ValueError("Username or email already exists.") from exc

    async def login(self, payload: AuthLoginRequest) -> AuthTokenPairResponse:
        user = await self.repository.get_user_by_username_or_email(
            payload.username_or_email
        )
        if user is None or not verify_password(payload.password, user.password_hash):
            raise PermissionError("Invalid credentials.")
        if not user.is_active:
            raise PermissionError("User is inactive.")
        return await self._issue_token_pair(user.id)

    async def refresh(self, refresh_token: str) -> AuthTokenPairResponse:
        payload = decode_token(
            refresh_token,
            secret=self.settings.auth_jwt_secret,
            algorithm=self.settings.auth_jwt_algorithm,
        )
        kind = payload.get("type")
        if kind != self.jwt_kind_refresh:
            raise PermissionError("Invalid token type for refresh.")

        user_id_raw = payload.get("sub")
        if not isinstance(user_id_raw, str):
            raise PermissionError("Invalid token subject.")
        user_id = UUID(user_id_raw)

        stored = await self.repository.get_refresh_token(hash_token(refresh_token))
        if stored is None:
            raise PermissionError("Refresh token not found.")
        if stored.revoked_at is not None:
            raise PermissionError("Refresh token revoked.")
        if self._as_utc(stored.expires_at) <= datetime.now(timezone.utc):
            raise PermissionError("Refresh token expired.")

        user = await self.repository.get_user_by_id(user_id)
        if user is None or not user.is_active:
            raise PermissionError("User not available.")

        await self.repository.revoke_refresh_token(
            stored,
            revoked_at=datetime.now(timezone.utc),
        )
        return await self._issue_token_pair(user_id=user_id)

    async def logout(self, refresh_token: str) -> None:
        hashed = hash_token(refresh_token)
        stored = await self.repository.get_refresh_token(hashed)
        if stored is None:
            return
        if stored.revoked_at is not None:
            return
        await self.repository.revoke_refresh_token(
            stored,
            revoked_at=datetime.now(timezone.utc),
        )

    async def get_user_from_access_token(self, access_token: str) -> User:
        payload = decode_token(
            access_token,
            secret=self.settings.auth_jwt_secret,
            algorithm=self.settings.auth_jwt_algorithm,
        )
        kind = payload.get("type")
        if kind != self.jwt_kind_access:
            raise PermissionError("Invalid token type for access.")

        user_id_raw = payload.get("sub")
        if not isinstance(user_id_raw, str):
            raise PermissionError("Invalid token subject.")

        user = await self.repository.get_user_by_id(UUID(user_id_raw))
        if user is None:
            raise PermissionError("User not found.")
        if not user.is_active:
            raise PermissionError("User is inactive.")
        return user

    async def _issue_token_pair(self, user_id: UUID) -> AuthTokenPairResponse:
        access = create_access_token(
            user_id=user_id,
            secret=self.settings.auth_jwt_secret,
            algorithm=self.settings.auth_jwt_algorithm,
            expires_minutes=self.settings.auth_access_token_ttl_minutes,
        )
        refresh = create_refresh_token(
            user_id=user_id,
            secret=self.settings.auth_jwt_secret,
            algorithm=self.settings.auth_jwt_algorithm,
            expires_days=self.settings.auth_refresh_token_ttl_days,
        )
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.settings.auth_refresh_token_ttl_days
        )
        await self.repository.create_refresh_token(
            AuthRefreshToken(
                user_id=user_id,
                token_hash=hash_token(refresh),
                expires_at=expires_at,
            )
        )
        return AuthTokenPairResponse(
            access_token=access,
            refresh_token=refresh,
            token_type=self.bearer_type,
        )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
