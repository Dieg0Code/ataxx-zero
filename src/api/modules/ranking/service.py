from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from api.db.enums import WinnerSide
from api.db.models import LeaderboardEntry, Rating, RatingEvent, Season
from api.modules.ranking.repository import RankingRepository
from api.modules.ranking.schemas import SeasonCreateRequest

ELO_K_FACTOR = 24.0
DEFAULT_RATING = 1200.0


class RankingService:
    def __init__(self, ranking_repository: RankingRepository) -> None:
        self.ranking_repository = ranking_repository

    async def create_season(self, payload: SeasonCreateRequest) -> Season:
        starts_at = payload.starts_at or datetime.now(timezone.utc)
        season = Season(
            name=payload.name,
            starts_at=starts_at,
            ends_at=payload.ends_at,
            is_active=payload.is_active,
        )
        return await self.ranking_repository.create_season(season)

    async def get_active_season(self) -> Season | None:
        return await self.ranking_repository.get_active_season()

    async def get_or_create_rating(self, user_id: UUID, season_id: UUID) -> Rating:
        rating = await self.ranking_repository.get_rating(user_id=user_id, season_id=season_id)
        if rating is not None:
            return rating
        new_rating = Rating(
            user_id=user_id,
            season_id=season_id,
            rating=DEFAULT_RATING,
        )
        return await self.ranking_repository.create_rating(new_rating)

    async def apply_rated_result(
        self,
        *,
        game_id: UUID,
        season_id: UUID,
        player1_id: UUID,
        player2_id: UUID,
        winner_side: WinnerSide,
    ) -> tuple[Rating, Rating]:
        p1 = await self.get_or_create_rating(player1_id, season_id)
        p2 = await self.get_or_create_rating(player2_id, season_id)

        expected_p1 = 1.0 / (1.0 + 10.0 ** ((p2.rating - p1.rating) / 400.0))
        expected_p2 = 1.0 - expected_p1

        if winner_side == WinnerSide.P1:
            actual_p1 = 1.0
            actual_p2 = 0.0
            p1.wins += 1
            p2.losses += 1
        elif winner_side == WinnerSide.P2:
            actual_p1 = 0.0
            actual_p2 = 1.0
            p1.losses += 1
            p2.wins += 1
        else:
            actual_p1 = 0.5
            actual_p2 = 0.5
            p1.draws += 1
            p2.draws += 1

        p1_before = p1.rating
        p2_before = p2.rating
        p1.rating = p1.rating + (ELO_K_FACTOR * (actual_p1 - expected_p1))
        p2.rating = p2.rating + (ELO_K_FACTOR * (actual_p2 - expected_p2))
        p1.games_played += 1
        p2.games_played += 1
        p1.updated_at = datetime.now(timezone.utc)
        p2.updated_at = datetime.now(timezone.utc)

        p1 = await self.ranking_repository.save_rating(p1)
        p2 = await self.ranking_repository.save_rating(p2)

        await self.ranking_repository.create_rating_event(
            RatingEvent(
                game_id=game_id,
                user_id=player1_id,
                season_id=season_id,
                rating_before=p1_before,
                rating_after=p1.rating,
                delta=p1.rating - p1_before,
            )
        )
        await self.ranking_repository.create_rating_event(
            RatingEvent(
                game_id=game_id,
                user_id=player2_id,
                season_id=season_id,
                rating_before=p2_before,
                rating_after=p2.rating,
                delta=p2.rating - p2_before,
            )
        )

        return p1, p2

    async def recompute_leaderboard(
        self,
        season_id: UUID,
        limit: int = 100,
    ) -> list[LeaderboardEntry]:
        safe_limit = max(1, min(limit, 500))
        ratings = await self.ranking_repository.list_ratings_for_season(
            season_id=season_id,
            limit=safe_limit,
        )
        computed_at = self.ranking_repository.now_utc()
        entries: list[LeaderboardEntry] = []
        for idx, rating in enumerate(ratings, start=1):
            total = rating.wins + rating.losses + rating.draws
            win_rate = (rating.wins / total) if total > 0 else 0.0
            entries.append(
                LeaderboardEntry(
                    season_id=season_id,
                    user_id=rating.user_id,
                    rank=idx,
                    rating=rating.rating,
                    wins=rating.wins,
                    losses=rating.losses,
                    draws=rating.draws,
                    win_rate=win_rate,
                    computed_at=computed_at,
                )
            )
        return await self.ranking_repository.replace_leaderboard_entries(
            season_id=season_id,
            entries=entries,
        )

    async def get_leaderboard(
        self,
        season_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[LeaderboardEntry]]:
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        total = await self.ranking_repository.count_leaderboard_entries(season_id=season_id)
        entries = await self.ranking_repository.get_leaderboard_entries(
            season_id=season_id,
            limit=safe_limit,
            offset=safe_offset,
        )
        if entries:
            return total, entries
        recomputed = await self.recompute_leaderboard(season_id=season_id, limit=safe_limit)
        total_after = await self.ranking_repository.count_leaderboard_entries(
            season_id=season_id
        )
        return total_after, recomputed[safe_offset : safe_offset + safe_limit]
