from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.session import get_session
from api.modules.gameplay.repository import GameRepository
from api.modules.gameplay.service import GameplayService
from api.modules.ranking.repository import RankingRepository
from api.modules.ranking.service import RankingService

SESSION_DEP = Depends(get_session)


def get_gameplay_service_dep(
    session: AsyncSession = SESSION_DEP,
) -> GameplayService:
    ranking_service = RankingService(ranking_repository=RankingRepository(session=session))
    return GameplayService(
        game_repository=GameRepository(session=session),
        ranking_service=ranking_service,
    )
