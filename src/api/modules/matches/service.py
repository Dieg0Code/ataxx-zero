from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import numpy as np

from agents.heuristic import heuristic_move
from api.db.enums import (
    AgentType,
    GameStatus,
    PlayerSide,
    TerminationReason,
    WinnerSide,
)
from api.db.models import BotProfile, Game, GameMove, User
from api.modules.matches.repository import MatchesRepository
from api.modules.matches.schemas import MatchCreateRequest, MatchMoveRequest
from api.modules.ranking.service import RankingService
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.constants import PLAYER_1
from game.serialization import board_from_state, board_to_state
from inference.service import InferenceService


class MatchesService:
    def __init__(
        self,
        repository: MatchesRepository,
        ranking_service: RankingService | None = None,
    ) -> None:
        self.repository = repository
        self.ranking_service = ranking_service

    async def create_match(self, payload: MatchCreateRequest, actor_user_id: UUID) -> Game:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if payload.player1_id is not None and payload.player1_id != actor_user_id:
            raise PermissionError("player1_id must match the authenticated user.")

        player2_agent = payload.player2_agent
        model_version_id = payload.model_version_id
        if payload.player2_id is not None:
            player2 = await self.repository.get_user(payload.player2_id)
            if player2 is None:
                raise LookupError(f"User not found: {payload.player2_id}")
            if player2.is_bot:
                profile = await self._get_enabled_bot_profile(player2.id)
                player2_agent = profile.agent_type
                if profile.agent_type == AgentType.MODEL and model_version_id is None:
                    model_version_id = player2.model_version_id

        game = Game(
            season_id=payload.season_id,
            queue_type=payload.queue_type,
            status=GameStatus.IN_PROGRESS,
            rated=payload.rated,
            player1_id=actor_user_id,
            player2_id=payload.player2_id,
            created_by_user_id=actor_user_id,
            player1_agent=payload.player1_agent,
            player2_agent=player2_agent,
            model_version_id=model_version_id,
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
        if actor_user_id in (game.player1_id, game.player2_id, game.created_by_user_id):
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

    async def advance_bot_turn(
        self,
        game_id: UUID,
        actor_user: User,
        inference_service: InferenceService | None = None,
    ) -> GameMove | None:
        game = await self.repository.get_game(game_id)
        if game is None:
            raise LookupError(f"Match not found: {game_id}")
        await self.ensure_can_view_match(game_id=game_id, actor_user=actor_user)
        if game.status == GameStatus.FINISHED:
            return None

        board = await self.get_current_board(game_id)
        bot_side = self._to_side(board.current_player)
        bot_user_id = game.player1_id if bot_side == PlayerSide.P1 else game.player2_id
        if bot_user_id is None:
            raise ValueError("Current turn is not assigned to a user.")

        bot_user = await self.repository.get_user(bot_user_id)
        if bot_user is None:
            raise LookupError(f"User not found: {bot_user_id}")
        if not bot_user.is_bot:
            raise ValueError("Current turn does not belong to a bot.")

        profile = await self._get_enabled_bot_profile(bot_user.id)
        legal_moves = board.get_valid_moves()

        mode: str
        action_idx: int
        value: float
        selected_move: tuple[int, int, int, int] | None
        if profile.agent_type == AgentType.HEURISTIC:
            level = profile.heuristic_level or "normal"
            if level not in {"easy", "normal", "hard"}:
                raise ValueError("Invalid heuristic_level for bot profile.")
            mode = f"heuristic_{level}"
            selected_move = heuristic_move(
                board=board,
                rng=np.random.default_rng(),
                level=level,
            )
            action_idx = (
                ACTION_SPACE.pass_index
                if selected_move is None
                else ACTION_SPACE.encode(selected_move)
            )
            value = 0.0
        elif profile.agent_type == AgentType.MODEL:
            if inference_service is None:
                raise RuntimeError("Inference service is required for model bots.")
            model_mode = profile.model_mode or "fast"
            if model_mode not in {"fast", "strong"}:
                raise ValueError("Invalid model_mode for bot profile.")
            prediction = inference_service.predict(board=board, mode=model_mode)
            selected_move = prediction.move
            mode = prediction.mode
            action_idx = prediction.action_idx
            value = prediction.value
        else:
            raise ValueError("Unsupported bot agent type.")

        if selected_move is None and legal_moves:
            raise ValueError("Bot selected pass while legal moves are available.")
        if selected_move is not None and selected_move not in legal_moves:
            raise ValueError("Bot selected an illegal move.")

        board_before = board_to_state(board)
        board.step(selected_move)
        board_after = board_to_state(board)
        ply = await self.repository.next_ply(game_id)

        stored = await self.repository.create_move(
            GameMove(
                game_id=game_id,
                ply=ply,
                player_side=bot_side,
                r1=selected_move[0] if selected_move else None,
                c1=selected_move[1] if selected_move else None,
                r2=selected_move[2] if selected_move else None,
                c2=selected_move[3] if selected_move else None,
                board_before=board_before,
                board_after=board_after,
                mode=mode,
                action_idx=action_idx,
                value=value,
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
        game.ended_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
            # Keep public leaderboard in sync with the latest rated result.
            await self.ranking_service.recompute_leaderboard(
                season_id=game.season_id,
                limit=500,
            )

    @staticmethod
    def _to_side(player: int) -> PlayerSide:
        return PlayerSide.P1 if player == PLAYER_1 else PlayerSide.P2

    async def _get_enabled_bot_profile(self, user_id: UUID) -> BotProfile:
        profile = await self.repository.get_bot_profile(user_id)
        if profile is None or not profile.enabled:
            raise ValueError("Bot profile is missing or disabled.")
        if profile.agent_type not in {AgentType.HEURISTIC, AgentType.MODEL}:
            raise ValueError("Bot profile agent_type must be heuristic or model.")
        return profile



