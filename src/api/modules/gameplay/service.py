from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.exc import DBAPIError

from api.db.enums import (
    AgentType,
    GameStatus,
    PlayerSide,
    QueueType,
    TerminationReason,
    WinnerSide,
)
from api.db.models import Game, GameMove, User
from api.modules.gameplay.repository import GameRepository
from api.modules.gameplay.schemas import GameCreateRequest
from api.modules.ranking.service import RankingService
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.serialization import board_to_state
from inference.service import InferenceResult

logger = logging.getLogger(__name__)


class MoveConflictError(ValueError):
    """Raised when a move collides with a newer persisted board or concurrent write."""


def _is_concurrent_move_db_error(exc: DBAPIError) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    # Multiple DB backends/drivers report write races with different messages.
    # Normalize to one conflict path so the API returns 409 instead of 500.
    conflict_signals = (
        "duplicate key value",
        "unique constraint",
        "uq_game_moves_game_ply",
        "deadlock detected",
        "could not serialize access",
        "serialization failure",
        "database is locked",
        "lock timeout",
    )
    return any(signal in message for signal in conflict_signals)


class GameplayService:
    def __init__(
        self,
        game_repository: GameRepository,
        ranking_service: RankingService | None = None,
    ) -> None:
        self.game_repository = game_repository
        self.ranking_service = ranking_service

    async def create_game(self, payload: GameCreateRequest, actor_user: User) -> Game:
        player1_id = payload.player1_id
        if actor_user.is_admin:
            if player1_id is None:
                player1_id = actor_user.id
        else:
            if player1_id is None:
                player1_id = actor_user.id
            elif player1_id != actor_user.id:
                can_create_bot_duel = await self._can_create_bot_duel(
                    player1_id=player1_id,
                    player2_id=payload.player2_id,
                    player1_agent=payload.player1_agent,
                    player2_agent=payload.player2_agent,
                )
                if not can_create_bot_duel:
                    raise PermissionError("player1_id must match authenticated user.")

        if player1_id is not None and payload.player2_id is not None and player1_id == payload.player2_id:
            raise ValueError("player1_id and player2_id must be different users.")

        is_ai_vs_ai = payload.player1_agent != AgentType.HUMAN and payload.player2_agent != AgentType.HUMAN
        if is_ai_vs_ai and (payload.rated or payload.queue_type == QueueType.RANKED):
            raise ValueError("IA vs IA solo admite modo casual (sin ranked/ELO).")

        if payload.player2_id is not None:
            player2 = await self.game_repository.get_user_by_id(payload.player2_id)
            if player2 is None:
                raise LookupError(f"User not found: {payload.player2_id}")

        game = Game(
            season_id=payload.season_id,
            queue_type=payload.queue_type,
            status=payload.status,
            rated=payload.rated,
            player1_id=player1_id,
            player2_id=payload.player2_id,
            created_by_user_id=actor_user.id,
            player1_agent=payload.player1_agent,
            player2_agent=payload.player2_agent,
            model_version_id=payload.model_version_id,
            source=payload.source,
            quality_score=payload.quality_score,
            is_training_eligible=payload.is_training_eligible,
        )
        return await self.game_repository.create(game)

    async def get_game(self, game_id: UUID) -> Game | None:
        return await self.game_repository.get_by_id(game_id)

    async def get_player_usernames(self, game: Game) -> tuple[str | None, str | None]:
        get_username = getattr(self.game_repository, "get_username_by_id", None)
        if get_username is None:
            return None, None
        player1_username = await get_username(game.player1_id)
        player2_username = await get_username(game.player2_id)
        return player1_username, player2_username

    async def list_games(
        self,
        *,
        limit: int,
        offset: int,
        actor_user: User,
        statuses: list[GameStatus] | None = None,
    ) -> tuple[int, list[Game]]:
        status_filter = statuses if statuses else None
        if actor_user.is_admin:
            if status_filter is None:
                total = await self.game_repository.count_all_games()
            else:
                total = await self.game_repository.count_all_games_by_status(
                    statuses=status_filter,
                )
            try:
                if status_filter is None:
                    games = await self.game_repository.list_recent(limit=limit, offset=offset)
                else:
                    games = await self.game_repository.list_recent_by_status(
                        statuses=status_filter,
                        limit=limit,
                        offset=offset,
                    )
            except ValueError:
                # Legacy/corrupt rows may fail enum parsing in some environments.
                games = await self._list_games_resilient(
                    limit=limit,
                    offset=offset,
                    user_id=None,
                    statuses=status_filter,
                )
            return total, games
        if status_filter is None:
            total = await self.game_repository.count_games_for_user(user_id=actor_user.id)
        else:
            total = await self.game_repository.count_games_for_user_by_status(
                user_id=actor_user.id,
                statuses=status_filter,
            )
        try:
            if status_filter is None:
                games = await self.game_repository.list_recent_for_user(
                    user_id=actor_user.id,
                    limit=limit,
                    offset=offset,
                )
            else:
                games = await self.game_repository.list_recent_for_user_by_status(
                    user_id=actor_user.id,
                    statuses=status_filter,
                    limit=limit,
                    offset=offset,
                )
        except ValueError:
            games = await self._list_games_resilient(
                limit=limit,
                offset=offset,
                user_id=actor_user.id,
                statuses=status_filter,
            )
        return total, games

    async def _list_games_resilient(
        self,
        *,
        limit: int,
        offset: int,
        user_id: UUID | None,
        statuses: list[GameStatus] | None = None,
    ) -> list[Game]:
        if user_id is None:
            if statuses is None:
                candidate_ids = await self.game_repository.list_recent_ids(limit=limit, offset=offset)
            else:
                candidate_ids = await self.game_repository.list_recent_ids_by_status(
                    statuses=statuses,
                    limit=limit,
                    offset=offset,
                )
        else:
            if statuses is None:
                candidate_ids = await self.game_repository.list_recent_ids_for_user(
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                )
            else:
                candidate_ids = await self.game_repository.list_recent_ids_for_user_by_status(
                    user_id=user_id,
                    statuses=statuses,
                    limit=limit,
                    offset=offset,
                )

        games: list[Game] = []
        for game_id in candidate_ids:
            try:
                game = await self.game_repository.get_by_id(game_id)
            except ValueError:
                logger.warning("Skipping corrupt game row during resilient list.", extra={"game_id": str(game_id)})
                continue
            if game is not None:
                games.append(game)
        return games

    async def ensure_can_view_game(self, game_id: UUID, actor_user: User) -> Game:
        game = await self.game_repository.get_by_id(game_id)
        if game is None:
            raise LookupError(f"Game not found: {game_id}")
        if actor_user.is_admin:
            return game
        if actor_user.id in (game.player1_id, game.player2_id, game.created_by_user_id):
            return game
        raise PermissionError("Authenticated user is not allowed to view this game.")

    async def _can_create_bot_duel(
        self,
        *,
        player1_id: UUID,
        player2_id: UUID | None,
        player1_agent: AgentType,
        player2_agent: AgentType,
    ) -> bool:
        if player2_id is None:
            return False
        if player1_agent == AgentType.HUMAN or player2_agent == AgentType.HUMAN:
            return False
        player1 = await self.game_repository.get_user_by_id(player1_id)
        player2 = await self.game_repository.get_user_by_id(player2_id)
        if player1 is None or player2 is None:
            return False
        return bool(player1.is_bot and player2.is_bot)

    async def delete_game(self, game_id: UUID, actor_user: User) -> None:
        try:
            await self.ensure_can_view_game(game_id=game_id, actor_user=actor_user)
        except ValueError:
            access_ids = await self.game_repository.get_game_access_ids(game_id)
            if access_ids is None:
                raise LookupError(f"Game not found: {game_id}") from None
            if not actor_user.is_admin and actor_user.id not in access_ids:
                raise PermissionError("Authenticated user is not allowed to view this game.") from None
        try:
            deleted = await self.game_repository.delete_game(game_id)
        except ValueError:
            logger.warning(
                "Falling back to force_delete_game due to invalid/corrupt row.",
                extra={"game_id": str(game_id)},
            )
            deleted = await self.game_repository.force_delete_game(game_id)
        if not deleted:
            raise LookupError(f"Game not found: {game_id}")

    async def record_inference_move(
        self,
        game_id: UUID,
        board: AtaxxBoard,
        inference: InferenceResult,
        actor_user: User,
    ) -> tuple[GameMove, Game]:
        game = await self.ensure_can_view_game(game_id=game_id, actor_user=actor_user)

        ply = await self.game_repository.next_ply(game_id)
        side = PlayerSide.P1 if board.current_player == 1 else PlayerSide.P2

        r1: int | None = None
        c1: int | None = None
        r2: int | None = None
        c2: int | None = None
        if inference.move is not None:
            r1, c1, r2, c2 = inference.move

        board_before = board_to_state(board)
        scratch = board.copy()
        scratch.step(inference.move)
        board_after = board_to_state(scratch)

        move = GameMove(
            game_id=game_id,
            ply=ply,
            player_side=side,
            r1=r1,
            c1=c1,
            r2=r2,
            c2=c2,
            board_before=board_before,
            board_after=board_after,
            mode=inference.mode,
            action_idx=inference.action_idx,
            value=inference.value,
        )
        stored_move = await self.game_repository.create_move_uncommitted(move)
        should_apply_rated_result = await self._update_game_state(
            game=game,
            board_after=scratch,
            commit=False,
        )
        await self.game_repository.commit()
        if should_apply_rated_result:
            await self._apply_rated_result(game)
        return stored_move, game

    async def list_game_moves(self, game_id: UUID, limit: int = 200) -> list[GameMove]:
        game = await self.game_repository.get_by_id(game_id)
        if game is None:
            raise LookupError(f"Game not found: {game_id}")
        return await self.game_repository.list_moves(game_id=game_id, limit=limit)

    async def record_manual_move(
        self,
        game_id: UUID,
        board: AtaxxBoard,
        move: tuple[int, int, int, int],
        actor_user: User,
        mode: str = "manual",
    ) -> tuple[GameMove, Game]:
        game = await self.ensure_can_view_game(game_id=game_id, actor_user=actor_user)
        board_before = board_to_state(board)
        last_move = await self.game_repository.get_last_move(game_id)
        if (
            last_move is not None
            and isinstance(last_move.board_after, dict)
            and last_move.board_after != board_before
        ):
            raise MoveConflictError(
                "Board state is stale; refresh game state and retry the move."
            )

        legal_moves = board.get_valid_moves()
        if move not in legal_moves:
            raise ValueError("Illegal move for provided board state.")

        ply = await self.game_repository.next_ply(game_id)
        side = PlayerSide.P1 if board.current_player == 1 else PlayerSide.P2
        scratch = board.copy()
        scratch.step(move)
        board_after = board_to_state(scratch)

        r1, c1, r2, c2 = move
        try:
            stored_move = await self.game_repository.create_move_uncommitted(
                GameMove(
                    game_id=game_id,
                    ply=ply,
                    player_side=side,
                    r1=r1,
                    c1=c1,
                    r2=r2,
                    c2=c2,
                    board_before=board_before,
                    board_after=board_after,
                    mode=mode,
                    action_idx=ACTION_SPACE.encode(move),
                    value=0.0,
                )
            )
            should_apply_rated_result = await self._update_game_state(
                game=game,
                board_after=scratch,
                commit=False,
            )
            await self.game_repository.commit()
        except DBAPIError as exc:
            await self.game_repository.rollback()
            if _is_concurrent_move_db_error(exc):
                raise MoveConflictError(
                    "Concurrent move conflict; refresh board and retry."
                ) from exc
            raise
        if should_apply_rated_result:
            await self._apply_rated_result(game)
        return stored_move, game

    async def _update_game_state(
        self,
        game: Game,
        board_after: AtaxxBoard,
        *,
        commit: bool = True,
    ) -> bool:
        changed = False
        if game.started_at is None:
            game.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
            changed = True

        if not board_after.is_game_over():
            if changed:
                if commit:
                    await self.game_repository.save_game(game)
                else:
                    await self.game_repository.save_game_uncommitted(game)
            return False

        result = board_after.get_result()
        if result == 1:
            game.winner_side = WinnerSide.P1
            game.winner_user_id = game.player1_id
        elif result == -1:
            game.winner_side = WinnerSide.P2
            game.winner_user_id = game.player2_id
        else:
            game.winner_side = WinnerSide.DRAW
            game.winner_user_id = None

        game.status = GameStatus.FINISHED
        game.termination_reason = TerminationReason.NORMAL
        game.ended_at = datetime.now(timezone.utc).replace(tzinfo=None)
        if commit:
            await self.game_repository.save_game(game)
        else:
            await self.game_repository.save_game_uncommitted(game)
        return self._should_apply_rated_result(game)

    def _should_apply_rated_result(self, game: Game) -> bool:
        return (
            game.rated
            and game.season_id is not None
            and game.player1_id is not None
            and game.player2_id is not None
            and game.winner_side is not None
            and self.ranking_service is not None
        )

    async def _apply_rated_result(self, game: Game) -> None:
        if self.ranking_service is None:
            raise RuntimeError("Ranking service is required for rated result finalization.")
        if game.season_id is None or game.player1_id is None or game.player2_id is None or game.winner_side is None:
            raise RuntimeError("Rated game is missing required identifiers for rating application.")
        await self.ranking_service.apply_rated_result(
            game_id=game.id,
            season_id=game.season_id,
            player1_id=game.player1_id,
            player2_id=game.player2_id,
            winner_side=game.winner_side,
        )
        # Keep leaderboard_entry synchronized after each rated game result.
        await self.ranking_service.recompute_leaderboard(
            season_id=game.season_id,
            limit=500,
        )


