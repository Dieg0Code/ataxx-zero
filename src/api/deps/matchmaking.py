from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.matches.repository import MatchesRepository
from api.modules.matches.service import MatchesService
from api.modules.matchmaking.repository import MatchmakingRepository
from api.modules.matchmaking.service import MatchmakingService
from api.modules.ranking.repository import RankingRepository
from api.modules.ranking.service import RankingService

SESSION_DEP = Depends(get_session)


def get_matchmaking_service_dep(session: AsyncSession = SESSION_DEP) -> MatchmakingService:
    ranking_service = RankingService(ranking_repository=RankingRepository(session=session))
    matches_service = MatchesService(
        repository=MatchesRepository(session=session),
        ranking_service=ranking_service,
    )
    return MatchmakingService(
        repository=MatchmakingRepository(session=session),
        ranking_service=ranking_service,
        matches_service=matches_service,
    )
