from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    return pwd_context.verify(password, password_hash)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(
    *,
    user_id: UUID,
    secret: str,
    algorithm: str,
    expires_minutes: int,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def create_refresh_token(
    *,
    user_id: UUID,
    secret: str,
    algorithm: str,
    expires_days: int,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=expires_days)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_token(token: str, *, secret: str, algorithm: str) -> dict[str, object]:
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as exc:
        raise ValueError("Invalid or expired token.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid token payload.")
    return payload
