from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.identity.repository import UserRepository
from api.modules.identity.service import IdentityService

SESSION_DEP = Depends(get_session)


def get_identity_service_dep(session: AsyncSession = SESSION_DEP) -> IdentityService:
    return IdentityService(user_repository=UserRepository(session=session))
