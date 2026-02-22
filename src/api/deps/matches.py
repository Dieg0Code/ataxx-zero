from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.matches.repository import MatchesRepository
from api.modules.matches.service import MatchesService
from api.modules.ranking.repository import RankingRepository
from api.modules.ranking.service import RankingService

SESSION_DEP = Depends(get_session)


def get_matches_service_dep(
    session: AsyncSession = SESSION_DEP,
) -> MatchesService:
    ranking_service = RankingService(ranking_repository=RankingRepository(session=session))
    return MatchesService(
        repository=MatchesRepository(session=session),
        ranking_service=ranking_service,
    )
