from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from api.db.enums import (
    GameStatus,
    PlayerSide,
    TerminationReason,
    WinnerSide,
)
from api.db.models import Game, GameMove, User
from api.modules.matches.repository import MatchesRepository
from api.modules.matches.schemas import MatchCreateRequest, MatchMoveRequest
from api.modules.ranking.service import RankingService
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.constants import PLAYER_1
from game.serialization import board_from_state, board_to_state


class MatchesService:
    def __init__(
        self,
        repository: MatchesRepository,
        ranking_service: RankingService | None = None,
    ) -> None:
        self.repository = repository
        self.ranking_service = ranking_service

    async def create_match(self, payload: MatchCreateRequest, actor_user_id: UUID) -> Game:
        now = datetime.now(timezone.utc)
        if payload.player1_id is not None and payload.player1_id != actor_user_id:
            raise PermissionError("player1_id must match the authenticated user.")

        game = Game(
            season_id=payload.season_id,
            queue_type=payload.queue_type,
            status=GameStatus.IN_PROGRESS,
            rated=payload.rated,
            player1_id=actor_user_id,
            player2_id=payload.player2_id,
            player1_agent=payload.player1_agent,
            player2_agent=payload.player2_agent,
            model_version_id=payload.model_version_id,
            source=payload.source,
            is_training_eligible=payload.is_training_eligible,
            started_at=now,
        )
        return await self.repository.create_game(game)

    async def get_match(self, game_id: UUID) -> Game | None:
        return await self.repository.get_game(game_id)

    async def ensure_can_view_match(self, game_id: UUID, actor_user: User) -> None:
        game = await self.repository.get_game(game_id)
        if game is None:
            raise LookupError(f"Match not found: {game_id}")
        actor_user_id = getattr(actor_user, "id", None)
        is_admin = bool(getattr(actor_user, "is_admin", False))
        if is_admin:
            return
        if actor_user_id in (game.player1_id, game.player2_id):
            return
        raise PermissionError("Authenticated user is not allowed to view this match.")

    async def get_current_board(self, game_id: UUID) -> AtaxxBoard:
        game = await self.repository.get_game(game_id)
        if game is None:
            raise LookupError(f"Match not found: {game_id}")
        last_move = await self.repository.get_last_move(game_id)
        if last_move is None or last_move.board_after is None:
            return AtaxxBoard()
        return board_from_state(last_move.board_after)

    async def submit_move(
        self,
        game_id: UUID,
        request: MatchMoveRequest,
        actor_user_id: UUID,
    ) -> GameMove:
        game = await self.repository.get_game(game_id)
        if game is None:
            raise LookupError(f"Match not found: {game_id}")
        if game.status == GameStatus.FINISHED:
            raise ValueError("Match already finished.")

        actor_side = self._resolve_actor_side(game=game, actor_user_id=actor_user_id)

        board = await self.get_current_board(game_id)
        expected_side = self._to_side(board.current_player)
        if actor_side != expected_side:
            raise PermissionError(
                f"Invalid turn: expected {expected_side.value}, got {actor_side.value}."
            )

        legal_moves = board.get_valid_moves()
        move: tuple[int, int, int, int] | None = None
        if request.pass_turn:
            if legal_moves:
                raise ValueError("Pass is illegal when legal moves exist.")
            action_idx = ACTION_SPACE.pass_index
        else:
            if request.move is None:
                raise ValueError("move is required when pass_turn is false.")
            move = (
                request.move.r1,
                request.move.c1,
                request.move.r2,
                request.move.c2,
            )
            if move not in legal_moves:
                raise ValueError("Illegal move for current board.")
            action_idx = ACTION_SPACE.encode(move)

        board_before = board_to_state(board)
        board.step(move)
        board_after = board_to_state(board)

        ply = await self.repository.next_ply(game_id)
        stored = await self.repository.create_move(
            GameMove(
                game_id=game_id,
                ply=ply,
                player_side=actor_side,
                r1=move[0] if move else None,
                c1=move[1] if move else None,
                r2=move[2] if move else None,
                c2=move[3] if move else None,
                board_before=board_before,
                board_after=board_after,
                mode="authoritative",
                action_idx=action_idx,
                value=0.0,
            )
        )

        await self._update_game_terminal_state(game=game, board=board)
        return stored

    @staticmethod
    def _resolve_actor_side(game: Game, actor_user_id: UUID) -> PlayerSide:
        if game.player1_id is None or game.player2_id is None:
            raise ValueError(
                "Match players are not configured; cannot authorize move actor."
            )
        if actor_user_id == game.player1_id:
            return PlayerSide.P1
        if actor_user_id == game.player2_id:
            return PlayerSide.P2
        raise PermissionError("Authenticated user is not a participant in this match.")

    async def _update_game_terminal_state(self, game: Game, board: AtaxxBoard) -> None:
        if not board.is_game_over():
            if game.status != GameStatus.IN_PROGRESS:
                game.status = GameStatus.IN_PROGRESS
                await self.repository.save_game(game)
            return

        result = board.get_result()
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
        game.ended_at = datetime.now(timezone.utc)
        await self.repository.save_game(game)

        if (
            game.rated
            and game.season_id is not None
            and game.player1_id is not None
            and game.player2_id is not None
            and game.winner_side is not None
            and self.ranking_service is not None
        ):
            await self.ranking_service.apply_rated_result(
                game_id=game.id,
                season_id=game.season_id,
                player1_id=game.player1_id,
                player2_id=game.player2_id,
                winner_side=game.winner_side,
            )

    @staticmethod
    def _to_side(player: int) -> PlayerSide:
        return PlayerSide.P1 if player == PLAYER_1 else PlayerSide.P2
