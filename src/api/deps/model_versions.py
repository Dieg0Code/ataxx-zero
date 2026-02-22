from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.model_versions.repository import ModelVersionRepository
from api.modules.model_versions.service import ModelVersionService

SESSION_DEP = Depends(get_session)


def get_model_version_service_dep(
    session: AsyncSession = SESSION_DEP,
) -> ModelVersionService:
    return ModelVersionService(repository=ModelVersionRepository(session=session))
