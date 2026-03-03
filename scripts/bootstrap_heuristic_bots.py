from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import col, select

# Permite ejecutar el script desde la raiz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@dataclass(frozen=True)
class HeuristicBotSpec:
    username: str
    heuristic_level: str
    notes: str


BOT_SPECS: tuple[HeuristicBotSpec, ...] = (
    HeuristicBotSpec(
        username="ub_apexcore_v1",
        heuristic_level="apex",
        notes="Heuristica top-tier orientada a solidez tactica.",
    ),
    HeuristicBotSpec(
        username="ub_gambitshade_v1",
        heuristic_level="gambit",
        notes="Heuristica exotica agresiva con presion lateral.",
    ),
    HeuristicBotSpec(
        username="ub_sentinelloom_v1",
        heuristic_level="sentinel",
        notes="Heuristica exotica posicional con control de movilidad.",
    ),
)


async def _ensure_heuristic_bots() -> None:
    from agents.heuristic import DEFAULT_HEURISTIC_LEVEL, is_supported_heuristic_level
    from api.db.enums import AgentType, BotKind
    from api.db.models import BotProfile, User
    from api.db.session import get_engine, get_sessionmaker
    from api.modules.ranking.repository import RankingRepository
    from api.modules.ranking.service import RankingService

    invalid = [spec.heuristic_level for spec in BOT_SPECS if not is_supported_heuristic_level(spec.heuristic_level)]
    if invalid:
        raise RuntimeError(f"Unsupported levels in BOT_SPECS: {invalid}")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        ranking_service = RankingService(ranking_repository=RankingRepository(session=session))
        season = await ranking_service.get_active_season()
        if season is None:
            raise RuntimeError("No active season found. Run scripts/bootstrap_active_season.py first.")

        created_or_updated: list[User] = []
        for spec in BOT_SPECS:
            user_stmt = select(User).where(col(User.username) == spec.username)
            user = (await session.execute(user_stmt)).scalars().first()
            if user is None:
                user = User(
                    username=spec.username,
                    email=f"{spec.username}@bots.local",
                    is_active=True,
                    is_admin=False,
                    is_bot=True,
                    bot_kind=BotKind.HEURISTIC,
                    is_hidden_bot=False,
                    model_version_id=None,
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)

            user.is_active = True
            user.is_bot = True
            user.bot_kind = BotKind.HEURISTIC
            user.is_hidden_bot = False
            session.add(user)
            await session.commit()
            await session.refresh(user)

            profile_stmt = select(BotProfile).where(col(BotProfile.user_id) == user.id)
            profile = (await session.execute(profile_stmt)).scalars().first()
            if profile is None:
                profile = BotProfile(
                    user_id=user.id,
                    agent_type=AgentType.HEURISTIC,
                    heuristic_level=DEFAULT_HEURISTIC_LEVEL,
                    model_mode=None,
                    enabled=True,
                )

            profile.agent_type = AgentType.HEURISTIC
            profile.heuristic_level = spec.heuristic_level
            profile.model_mode = None
            profile.enabled = True
            session.add(profile)
            await session.commit()
            await session.refresh(profile)

            await ranking_service.get_or_create_rating(user.id, season.id)
            created_or_updated.append(user)

        leaderboard = await ranking_service.recompute_leaderboard(season_id=season.id, limit=500)
        rank_by_user_id = {entry.user_id: entry.rank for entry in leaderboard}

        print("Heuristic bots ready:")
        print(f"  season_id={season.id}")
        for spec in BOT_SPECS:
            user = next((item for item in created_or_updated if item.username == spec.username), None)
            if user is None:
                continue
            rating = await ranking_service.get_or_create_rating(user.id, season.id)
            rank = rank_by_user_id.get(user.id)
            print(
                "  "
                f"username={user.username} "
                f"level={spec.heuristic_level} "
                f"rating={rating.rating:.1f} "
                f"rank={rank}"
            )

    await get_engine().dispose()


def main() -> None:
    asyncio.run(_ensure_heuristic_bots())


if __name__ == "__main__":
    main()
