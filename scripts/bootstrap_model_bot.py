from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlmodel import col, select

# Permite ejecutar el script desde la raiz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea/actualiza una cuenta bot de modelo y la deja visible en ladder.",
    )
    parser.add_argument("--username", default="ub_bogonet_v0")
    parser.add_argument("--model-version-name", default="ub_bogonet_v0_iter039")
    parser.add_argument("--hf-repo-id", default="dieg0code/ataxx-zero")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--checkpoint-uri", default="hf://dieg0code/ataxx-zero/model_iter_039.pt")
    parser.add_argument("--onnx-uri", default="")
    parser.add_argument("--model-mode", choices=["fast", "strong"], default="fast")
    parser.add_argument("--activate-version", action="store_true")
    return parser.parse_args()


async def _ensure_model_bot(args: argparse.Namespace) -> None:
    from api.db.enums import AgentType, BotKind
    from api.db.models import BotProfile, ModelVersion, User
    from api.db.session import get_engine, get_sessionmaker
    from api.modules.ranking.repository import RankingRepository
    from api.modules.ranking.service import RankingService

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        version_stmt = select(ModelVersion).where(
            col(ModelVersion.name) == args.model_version_name
        )
        version = (await session.execute(version_stmt)).scalars().first()
        if version is None:
            version = ModelVersion(
                name=args.model_version_name,
                hf_repo_id=args.hf_repo_id,
                hf_revision=args.hf_revision,
                checkpoint_uri=args.checkpoint_uri,
                onnx_uri=(args.onnx_uri or None),
                is_active=bool(args.activate_version),
                notes="Bot bootstrap script.",
            )
            session.add(version)
            await session.commit()
            await session.refresh(version)
        else:
            version.hf_repo_id = args.hf_repo_id
            version.hf_revision = args.hf_revision
            version.checkpoint_uri = args.checkpoint_uri
            version.onnx_uri = args.onnx_uri or None
            if args.activate_version:
                version.is_active = True
            session.add(version)
            await session.commit()
            await session.refresh(version)

        if args.activate_version:
            await session.execute(
                # Keep one global active version when requested explicitly.
                ModelVersion.__table__.update()
                .where(col(ModelVersion.id) != version.id)
                .values(is_active=False)
            )
            await session.execute(
                ModelVersion.__table__.update()
                .where(col(ModelVersion.id) == version.id)
                .values(is_active=True)
            )
            await session.commit()

        user_stmt = select(User).where(col(User.username) == args.username)
        user = (await session.execute(user_stmt)).scalars().first()
        if user is None:
            user = User(
                username=args.username,
                email=f"{args.username}@bots.local",
                is_active=True,
                is_admin=False,
                is_bot=True,
                bot_kind=BotKind.MODEL,
                is_hidden_bot=False,
                model_version_id=version.id,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

        user.is_active = True
        user.is_bot = True
        user.is_hidden_bot = False
        user.bot_kind = BotKind.MODEL
        user.model_version_id = version.id
        session.add(user)
        await session.commit()
        await session.refresh(user)

        profile_stmt = select(BotProfile).where(col(BotProfile.user_id) == user.id)
        profile = (await session.execute(profile_stmt)).scalars().first()
        if profile is None:
            profile = BotProfile(
                user_id=user.id,
                agent_type=AgentType.MODEL,
                model_mode=args.model_mode,
                enabled=True,
            )
        profile.agent_type = AgentType.MODEL
        profile.model_mode = args.model_mode
        profile.heuristic_level = None
        profile.enabled = True
        session.add(profile)
        await session.commit()
        await session.refresh(profile)

        ranking_service = RankingService(ranking_repository=RankingRepository(session=session))
        season = await ranking_service.get_active_season()
        if season is None:
            raise RuntimeError("No active season found. Run scripts/bootstrap_active_season.py first.")
        rating = await ranking_service.get_or_create_rating(user.id, season.id)
        entries = await ranking_service.recompute_leaderboard(season_id=season.id, limit=500)
        rank = next((entry.rank for entry in entries if entry.user_id == user.id), None)

        print("Model bot ready:")
        print(f"  user_id={user.id}")
        print(f"  username={user.username}")
        print(f"  model_version_id={version.id}")
        print(f"  checkpoint_uri={version.checkpoint_uri}")
        print(f"  model_mode={profile.model_mode}")
        print(f"  season_id={season.id}")
        print(f"  rating={rating.rating:.1f}")
        print(f"  rank={rank}")

    await get_engine().dispose()


def main() -> None:
    args = _parse_args()
    asyncio.run(_ensure_model_bot(args))


if __name__ == "__main__":
    main()
