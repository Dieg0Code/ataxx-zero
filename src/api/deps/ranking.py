from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.ranking.repository import RankingRepository
from api.modules.ranking.service import RankingService

SESSION_DEP = Depends(get_session)


def get_ranking_service_dep(session: AsyncSession = SESSION_DEP) -> RankingService:
    return RankingService(ranking_repository=RankingRepository(session=session))
