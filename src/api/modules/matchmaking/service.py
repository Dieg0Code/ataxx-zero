from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from api.db.enums import AgentType, QueueEntryStatus, QueueType
from api.db.models import QueueEntry, User
from api.modules.matches.schemas import MatchCreateRequest
from api.modules.matches.service import MatchesService
from api.modules.matchmaking.repository import MatchmakingRepository
from api.modules.matchmaking.schemas import (
    QueueDecisionResponse,
    QueueJoinResponse,
    QueueLeaveResponse,
    QueueStatusResponse,
)
from api.modules.ranking.service import RankingService

MATCH_RANGE_POINTS = 120.0

MatchedWith = Literal["human", "bot"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MatchmakingService:
    def __init__(
        self,
        repository: MatchmakingRepository,
        ranking_service: RankingService,
        matches_service: MatchesService,
        rng: random.Random | None = None,
    ) -> None:
        self.repository = repository
        self.ranking_service = ranking_service
        self.matches_service = matches_service
        self.rng = rng or random.Random()  # noqa: S311

    @staticmethod
    def _status_value(status: QueueEntryStatus) -> Literal["waiting", "matched", "canceled"]:
        mapping: dict[QueueEntryStatus, Literal["waiting", "matched", "canceled"]] = {
            QueueEntryStatus.WAITING: "waiting",
            QueueEntryStatus.MATCHED: "matched",
            QueueEntryStatus.CANCELED: "canceled",
        }
        return mapping[status]

    async def join_ranked_queue(self, *, actor_user: User) -> QueueJoinResponse:
        season = await self.ranking_service.get_active_season()
        if season is None:
            raise LookupError("No active season found.")

        rating = await self.ranking_service.get_or_create_rating(actor_user.id, season.id)
        now = utcnow()
        entry = await self.repository.get_entry_for_user(season_id=season.id, user_id=actor_user.id)
        if entry is None:
            entry = QueueEntry(
                season_id=season.id,
                user_id=actor_user.id,
                rating_snapshot=rating.rating,
                status=QueueEntryStatus.WAITING,
                created_at=now,
                updated_at=now,
            )
            entry = await self.repository.create_entry(entry)
        else:
            entry.status = QueueEntryStatus.WAITING
            entry.matched_game_id = None
            entry.matched_at = None
            entry.canceled_at = None
            entry.rating_snapshot = rating.rating
            entry.updated_at = now
            entry = await self.repository.save_entry(entry)

        matched_with, game_id = await self._try_match_entry(entry=entry)
        refreshed = await self.repository.get_entry_by_id(entry.id)
        if refreshed is None:
            raise LookupError("Queue entry not found after join.")
        return QueueJoinResponse(
            queue_id=refreshed.id,
            status=self._status_value(refreshed.status),
            season_id=refreshed.season_id,
            game_id=game_id,
            matched_with=matched_with,
            created_at=refreshed.created_at,
            updated_at=refreshed.updated_at,
        )

    async def get_status(self, *, actor_user: User) -> QueueStatusResponse:
        season = await self.ranking_service.get_active_season()
        if season is None:
            return QueueStatusResponse(
                queue_id=None,
                status="idle",
                season_id=None,
                game_id=None,
            )
        entry = await self.repository.get_entry_for_user(season_id=season.id, user_id=actor_user.id)
        if entry is None:
            return QueueStatusResponse(
                queue_id=None,
                status="idle",
                season_id=season.id,
                game_id=None,
            )

        matched_with: MatchedWith | None = None
        if entry.status == QueueEntryStatus.WAITING:
            matched_with, _game_id = await self._try_match_entry(entry=entry)
            if _game_id is not None:
                refreshed = await self.repository.get_entry_by_id(entry.id)
                if refreshed is not None:
                    entry = refreshed

        return QueueStatusResponse(
            queue_id=entry.id,
            status=self._status_value(entry.status),
            season_id=entry.season_id,
            game_id=entry.matched_game_id,
            matched_with=matched_with,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )

    async def leave_queue(self, *, actor_user: User) -> QueueLeaveResponse:
        season = await self.ranking_service.get_active_season()
        if season is None:
            return QueueLeaveResponse(left_queue=False, status="idle")

        left = await self.repository.cancel_waiting_entry(
            season_id=season.id,
            user_id=actor_user.id,
            now=utcnow(),
        )
        return QueueLeaveResponse(left_queue=left, status="canceled" if left else "idle")

    async def accept_current_match(self, *, actor_user: User) -> QueueDecisionResponse:
        season = await self.ranking_service.get_active_season()
        if season is None:
            raise LookupError("No active season found.")
        entry = await self.repository.get_matched_entry_for_user(
            season_id=season.id,
            user_id=actor_user.id,
        )
        if entry is None or entry.matched_game_id is None:
            raise ValueError("No matched game pending confirmation.")
        return QueueDecisionResponse(
            decision="accepted",
            queue_id=entry.id,
            status=self._status_value(entry.status),
            game_id=entry.matched_game_id,
            updated_at=entry.updated_at,
        )

    async def reject_current_match(self, *, actor_user: User) -> QueueDecisionResponse:
        season = await self.ranking_service.get_active_season()
        if season is None:
            raise LookupError("No active season found.")
        entry = await self.repository.get_matched_entry_for_user(
            season_id=season.id,
            user_id=actor_user.id,
        )
        if entry is None or entry.matched_game_id is None:
            raise ValueError("No matched game pending confirmation.")

        move_count = await self.repository.count_game_moves(game_id=entry.matched_game_id)
        if move_count > 0:
            raise ValueError("Match already started and cannot be rejected.")

        now = utcnow()
        game_id = entry.matched_game_id
        matched_entries = await self.repository.list_matched_entries_for_game(
            season_id=season.id,
            game_id=game_id,
        )
        for matched_entry in matched_entries:
            if matched_entry.user_id == actor_user.id:
                matched_entry.status = QueueEntryStatus.CANCELED
                matched_entry.canceled_at = now
            else:
                matched_entry.status = QueueEntryStatus.WAITING
                matched_entry.canceled_at = None
            matched_entry.matched_game_id = None
            matched_entry.matched_at = None
            matched_entry.updated_at = now
            await self.repository.save_entry(matched_entry)

        await self.repository.delete_game_cascade(game_id=game_id)

        refreshed = await self.repository.get_entry_by_id(entry.id)
        if refreshed is None:
            raise LookupError("Queue entry not found after reject.")
        return QueueDecisionResponse(
            decision="rejected",
            queue_id=refreshed.id,
            status=self._status_value(refreshed.status),
            game_id=refreshed.matched_game_id,
            updated_at=refreshed.updated_at,
        )

    async def _try_match_entry(
        self,
        *,
        entry: QueueEntry,
    ) -> tuple[MatchedWith | None, UUID | None]:
        if entry.status != QueueEntryStatus.WAITING:
            return None, entry.matched_game_id

        min_rating = entry.rating_snapshot - MATCH_RANGE_POINTS
        max_rating = entry.rating_snapshot + MATCH_RANGE_POINTS

        human_candidates = await self.repository.list_waiting_human_entries(
            season_id=entry.season_id,
            user_id=entry.user_id,
            min_rating=min_rating,
            max_rating=max_rating,
            limit=50,
        )

        bot_candidates: list[tuple[User, float]] = []
        for bot_user, _profile in await self.repository.list_playable_bots(limit=200):
            bot_rating = await self.ranking_service.get_or_create_rating(bot_user.id, entry.season_id)
            if min_rating <= bot_rating.rating <= max_rating:
                bot_candidates.append((bot_user, bot_rating.rating))

        candidate_type: str | None = None
        if human_candidates and bot_candidates:
            candidate_type = "human" if self.rng.random() < 0.5 else "bot"
        elif human_candidates:
            candidate_type = "human"
        elif bot_candidates:
            candidate_type = "bot"
        else:
            return None, None

        now = utcnow()
        if candidate_type == "human":
            opponent = human_candidates[0]
            game = await self.matches_service.create_match(
                MatchCreateRequest(
                    season_id=entry.season_id,
                    queue_type=QueueType.RANKED,
                    rated=True,
                    player2_id=opponent.user_id,
                    player1_agent=AgentType.HUMAN,
                    player2_agent=AgentType.HUMAN,
                    source="human",
                    is_training_eligible=False,
                ),
                actor_user_id=entry.user_id,
            )
            entry.status = QueueEntryStatus.MATCHED
            entry.matched_game_id = game.id
            entry.matched_at = now
            entry.updated_at = now
            await self.repository.save_entry(entry)

            opponent.status = QueueEntryStatus.MATCHED
            opponent.matched_game_id = game.id
            opponent.matched_at = now
            opponent.updated_at = now
            await self.repository.save_entry(opponent)
            return "human", game.id

        last_bot_opponent_id = await self.repository.get_last_ranked_bot_opponent_user_id(
            season_id=entry.season_id,
            user_id=entry.user_id,
        )
        opponent_bot = self._select_bot_candidate(
            player_rating=entry.rating_snapshot,
            candidates=bot_candidates,
            last_bot_opponent_id=last_bot_opponent_id,
        )
        game = await self.matches_service.create_match(
            MatchCreateRequest(
                season_id=entry.season_id,
                queue_type=QueueType.RANKED,
                rated=True,
                player2_id=opponent_bot.id,
                player1_agent=AgentType.HUMAN,
                player2_agent=AgentType.HUMAN,
                source="human",
                is_training_eligible=False,
            ),
            actor_user_id=entry.user_id,
        )
        entry.status = QueueEntryStatus.MATCHED
        entry.matched_game_id = game.id
        entry.matched_at = now
        entry.updated_at = now
        await self.repository.save_entry(entry)
        return "bot", game.id

    def _select_bot_candidate(
        self,
        *,
        player_rating: float,
        candidates: list[tuple[User, float]],
        last_bot_opponent_id: UUID | None,
    ) -> User:
        if len(candidates) == 1:
            return candidates[0][0]

        weights: list[float] = []
        for bot_user, bot_rating in candidates:
            diff = abs(bot_rating - player_rating)
            # Closer bot rating gets higher weight; keep floor to preserve diversity.
            proximity_weight = max(0.05, 1.0 - (diff / MATCH_RANGE_POINTS))
            if last_bot_opponent_id is not None and bot_user.id == last_bot_opponent_id:
                # Soft penalty only: avoid hard exclusion and keep sampling fair.
                proximity_weight *= 0.35
            weights.append(proximity_weight)

        total_weight = sum(weights)
        if total_weight <= 0:
            return self.rng.choice([bot for bot, _ in candidates])

        roll = self.rng.random() * total_weight
        cumulative = 0.0
        for (bot_user, _), weight in zip(candidates, weights, strict=True):
            cumulative += weight
            if roll <= cumulative:
                return bot_user
        return candidates[-1][0]
