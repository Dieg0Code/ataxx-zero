from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.training_samples.repository import TrainingSamplesRepository
from api.modules.training_samples.service import TrainingSamplesService

SESSION_DEP = Depends(get_session)


def get_training_samples_service_dep(
    session: AsyncSession = SESSION_DEP,
) -> TrainingSamplesService:
    return TrainingSamplesService(repository=TrainingSamplesRepository(session=session))
