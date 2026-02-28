from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple
from uuid import UUID

from api.db.enums import WinnerSide
from api.db.models import LeaderboardEntry, Rating, RatingEvent, Season
from api.modules.ranking.repository import RankingRepository
from api.modules.ranking.schemas import SeasonCreateRequest

ELO_K_FACTOR = 24.0
ELO_DELTA_CAP = 30.0
DEFAULT_RATING = 1200.0
LEAGUE_NAMES = ("Protocol", "Kernel", "Root")
DIVISION_NAMES = ("III", "II", "I")
RATING_PER_DIVISION = 100.0
TOTAL_DIVISIONS = 9
MAJOR_PROMO_NAMES = {
    ("Protocol", "Kernel"): "Kernel Glitch",
    ("Kernel", "Root"): "Root Access",
}
TOP_PRESTIGE_TITLE = "Singularity"


class LadderSnapshot(NamedTuple):
    league: str
    division: str
    lp: int


class LadderTransition(NamedTuple):
    before: LadderSnapshot
    after: LadderSnapshot
    promoted: bool
    demoted: bool
    major_promo: str | None


class RankingService:
    TOP_PRESTIGE_TITLE = TOP_PRESTIGE_TITLE

    def __init__(self, ranking_repository: RankingRepository) -> None:
        self.ranking_repository = ranking_repository

    async def create_season(self, payload: SeasonCreateRequest) -> Season:
        starts_at = payload.starts_at or datetime.now(timezone.utc).replace(tzinfo=None)
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
        p1_delta = self._clamp_delta(ELO_K_FACTOR * (actual_p1 - expected_p1))
        p2_delta = self._clamp_delta(ELO_K_FACTOR * (actual_p2 - expected_p2))
        p1.rating = p1.rating + p1_delta
        p2.rating = p2.rating + p2_delta
        p1.games_played += 1
        p2.games_played += 1
        p1.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        p2.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        p1 = await self.ranking_repository.save_rating(p1)
        p2 = await self.ranking_repository.save_rating(p2)

        p1_transition = self.build_ladder_transition(p1_before, p1.rating)
        p2_transition = self.build_ladder_transition(p2_before, p2.rating)

        await self.ranking_repository.create_rating_event(
            RatingEvent(
                game_id=game_id,
                user_id=player1_id,
                season_id=season_id,
                rating_before=p1_before,
                rating_after=p1.rating,
                delta=p1.rating - p1_before,
                before_league=p1_transition.before.league,
                before_division=p1_transition.before.division,
                before_lp=p1_transition.before.lp,
                after_league=p1_transition.after.league,
                after_division=p1_transition.after.division,
                after_lp=p1_transition.after.lp,
                transition_type=(
                    "promotion"
                    if p1_transition.promoted
                    else ("demotion" if p1_transition.demoted else "stable")
                ),
                major_promo_name=p1_transition.major_promo,
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
                before_league=p2_transition.before.league,
                before_division=p2_transition.before.division,
                before_lp=p2_transition.before.lp,
                after_league=p2_transition.after.league,
                after_division=p2_transition.after.division,
                after_lp=p2_transition.after.lp,
                transition_type=(
                    "promotion"
                    if p2_transition.promoted
                    else ("demotion" if p2_transition.demoted else "stable")
                ),
                major_promo_name=p2_transition.major_promo,
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
        competitor_filter: str = "all",
        username_query: str | None = None,
    ) -> tuple[int, list[LeaderboardEntry]]:
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        total = await self.ranking_repository.count_leaderboard_entries(
            season_id=season_id,
            competitor_filter=competitor_filter,
            username_query=username_query,
        )
        entries = await self.ranking_repository.get_leaderboard_entries(
            season_id=season_id,
            limit=safe_limit,
            offset=safe_offset,
            competitor_filter=competitor_filter,
            username_query=username_query,
        )
        if entries:
            return total, entries
        # Recompute only for the canonical unfiltered leaderboard view.
        if competitor_filter != "all" or username_query is not None:
            return total, []
        recomputed = await self.recompute_leaderboard(season_id=season_id, limit=safe_limit)
        total_after = await self.ranking_repository.count_leaderboard_entries(
            season_id=season_id,
            competitor_filter=competitor_filter,
            username_query=username_query,
        )
        return total_after, recomputed[safe_offset : safe_offset + safe_limit]

    async def get_usernames_for_user_ids(self, user_ids: list[UUID]) -> dict[UUID, str]:
        return await self.ranking_repository.get_usernames_for_user_ids(user_ids)

    async def get_user_public_info_for_user_ids(
        self,
        user_ids: list[UUID],
    ) -> dict[UUID, tuple[str, bool, str | None]]:
        return await self.ranking_repository.get_user_public_info_for_user_ids(user_ids)

    async def get_rating_events(
        self,
        *,
        user_id: UUID,
        season_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[RatingEvent]]:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        total = await self.ranking_repository.count_rating_events(
            user_id=user_id,
            season_id=season_id,
        )
        events = await self.ranking_repository.list_rating_events(
            user_id=user_id,
            season_id=season_id,
            limit=safe_limit,
            offset=safe_offset,
        )
        return total, events

    async def get_latest_rating_events_for_user_ids(
        self,
        season_id: UUID,
        user_ids: list[UUID],
    ) -> dict[UUID, RatingEvent]:
        return await self.ranking_repository.get_latest_rating_events_for_user_ids(
            season_id=season_id,
            user_ids=user_ids,
        )

    @staticmethod
    def _clamp_delta(delta: float) -> float:
        return max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, delta))

    @staticmethod
    def build_league_snapshot(rating: float) -> tuple[str, str, int]:
        if rating <= DEFAULT_RATING:
            return LEAGUE_NAMES[0], DIVISION_NAMES[0], 0

        progress = rating - DEFAULT_RATING
        division_idx = int(progress // RATING_PER_DIVISION)

        if division_idx >= TOTAL_DIVISIONS:
            return LEAGUE_NAMES[-1], DIVISION_NAMES[-1], 99

        league_idx = division_idx // 3
        division_in_league = division_idx % 3
        lp = int(progress - (division_idx * RATING_PER_DIVISION))
        lp = max(0, min(99, lp))
        return (
            LEAGUE_NAMES[league_idx],
            DIVISION_NAMES[division_in_league],
            lp,
        )

    @staticmethod
    def next_major_promo_name(league: str, division: str) -> str | None:
        if division != "I":
            return None
        if league == "Protocol":
            return MAJOR_PROMO_NAMES[("Protocol", "Kernel")]
        if league == "Kernel":
            return MAJOR_PROMO_NAMES[("Kernel", "Root")]
        return None

    @staticmethod
    def _snapshot_division_index(snapshot: LadderSnapshot) -> int:
        league_idx = LEAGUE_NAMES.index(snapshot.league)
        division_idx = DIVISION_NAMES.index(snapshot.division)
        return league_idx * 3 + division_idx

    @classmethod
    def build_ladder_transition(
        cls,
        before_rating: float,
        after_rating: float,
    ) -> LadderTransition:
        before_raw = cls.build_league_snapshot(before_rating)
        after_raw = cls.build_league_snapshot(after_rating)
        before = LadderSnapshot(*before_raw)
        after = LadderSnapshot(*after_raw)
        before_idx = cls._snapshot_division_index(before)
        after_idx = cls._snapshot_division_index(after)
        promoted = after_idx > before_idx
        demoted = after_idx < before_idx
        major_promo = None
        if promoted and before.division == "I":
            major_promo = cls.next_major_promo_name(before.league, before.division)
        return LadderTransition(
            before=before,
            after=after,
            promoted=promoted,
            demoted=demoted,
            major_promo=major_promo,
        )



