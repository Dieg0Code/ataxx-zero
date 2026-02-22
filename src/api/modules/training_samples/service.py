from __future__ import annotations

import io
import json
from uuid import UUID

import numpy as np

from api.db.enums import GameSource, GameStatus, PlayerSide, SampleSplit, WinnerSide
from api.db.models import TrainingSample
from api.modules.training_samples.repository import TrainingSamplesRepository
from api.modules.training_samples.schemas import TrainingSampleCreateRequest


class TrainingSamplesService:
    def __init__(self, repository: TrainingSamplesRepository) -> None:
        self.repository = repository

    async def create_sample(self, payload: TrainingSampleCreateRequest) -> TrainingSample:
        game = await self.repository.get_game(payload.game_id)
        if game is None:
            raise LookupError(f"Game not found: {payload.game_id}")
        sample = TrainingSample(**payload.model_dump())
        return await self.repository.create(sample)

    async def get_sample(self, sample_id: UUID) -> TrainingSample | None:
        return await self.repository.get_by_id(sample_id)

    async def list_samples(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
    ) -> tuple[int, list[TrainingSample]]:
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        total = await self.repository.count_samples(split=split, game_id=game_id)
        rows = await self.repository.list_samples(
            limit=safe_limit,
            offset=safe_offset,
            split=split,
            game_id=game_id,
        )
        return total, rows

    async def stats(
        self,
        *,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
    ) -> tuple[int, dict[SampleSplit, int], dict[GameSource, int]]:
        samples = await self.repository.list_all_samples(split=split, game_id=game_id)
        by_split: dict[SampleSplit, int] = {}
        by_source: dict[GameSource, int] = {}
        for sample in samples:
            by_split[sample.split] = by_split.get(sample.split, 0) + 1
            by_source[sample.source] = by_source.get(sample.source, 0) + 1
        return len(samples), by_split, by_source

    async def export_ndjson(
        self,
        *,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
        limit: int = 5000,
    ) -> str:
        safe_limit = max(1, min(limit, 50_000))
        samples = await self.repository.list_samples(
            limit=safe_limit,
            split=split,
            game_id=game_id,
        )
        lines: list[str] = []
        for sample in samples:
            lines.append(
                json.dumps(
                    {
                        "id": str(sample.id),
                        "game_id": str(sample.game_id),
                        "move_id": str(sample.move_id) if sample.move_id else None,
                        "ply": sample.ply,
                        "player_side": sample.player_side.value,
                        "observation": sample.observation,
                        "policy_target": sample.policy_target,
                        "value_target": sample.value_target,
                        "sample_weight": sample.sample_weight,
                        "split": sample.split.value,
                        "source": sample.source.value,
                        "created_at": sample.created_at.isoformat(),
                    },
                    ensure_ascii=True,
                )
            )
        return "\n".join(lines) + ("\n" if lines else "")

    async def export_npz(
        self,
        *,
        split: SampleSplit | None = None,
        game_id: UUID | None = None,
        limit: int = 5000,
    ) -> bytes:
        safe_limit = max(1, min(limit, 50_000))
        samples = await self.repository.list_samples(
            limit=safe_limit,
            split=split,
            game_id=game_id,
        )

        sample_ids = np.array([str(sample.id) for sample in samples], dtype=np.str_)
        game_ids = np.array([str(sample.game_id) for sample in samples], dtype=np.str_)
        move_ids = np.array(
            [str(sample.move_id) if sample.move_id else "" for sample in samples],
            dtype=np.str_,
        )
        plys = np.array([sample.ply for sample in samples], dtype=np.int32)
        player_sides = np.array(
            [sample.player_side.value for sample in samples],
            dtype=np.str_,
        )
        observations_json = np.array(
            [json.dumps(sample.observation, ensure_ascii=True) for sample in samples],
            dtype=np.str_,
        )
        policy_targets_json = np.array(
            [json.dumps(sample.policy_target, ensure_ascii=True) for sample in samples],
            dtype=np.str_,
        )
        value_targets = np.array(
            [sample.value_target for sample in samples],
            dtype=np.float32,
        )
        sample_weights = np.array(
            [sample.sample_weight for sample in samples],
            dtype=np.float32,
        )
        splits = np.array([sample.split.value for sample in samples], dtype=np.str_)
        sources = np.array([sample.source.value for sample in samples], dtype=np.str_)
        created_at = np.array(
            [sample.created_at.isoformat() for sample in samples],
            dtype=np.str_,
        )

        output = io.BytesIO()
        np.savez_compressed(
            output,
            sample_ids=sample_ids,
            game_ids=game_ids,
            move_ids=move_ids,
            plys=plys,
            player_sides=player_sides,
            observations_json=observations_json,
            policy_targets_json=policy_targets_json,
            value_targets=value_targets,
            sample_weights=sample_weights,
            splits=splits,
            sources=sources,
            created_at=created_at,
        )
        return output.getvalue()

    async def ingest_from_game(
        self,
        *,
        game_id: UUID,
        split: SampleSplit = SampleSplit.TRAIN,
        source: GameSource = GameSource.SELF_PLAY,
        overwrite: bool = False,
    ) -> list[TrainingSample]:
        game = await self.repository.get_game(game_id)
        if game is None:
            raise LookupError(f"Game not found: {game_id}")
        if game.status != GameStatus.FINISHED or game.winner_side is None:
            raise ValueError(
                "Game must be finished and have winner_side before ingestion."
            )

        moves = await self.repository.list_game_moves(game_id)
        if not moves:
            raise ValueError("Game has no moves to ingest.")

        if overwrite:
            await self.repository.delete_for_game(game_id=game_id)

        samples: list[TrainingSample] = []
        for move in moves:
            if move.board_before is None:
                continue
            value_target = self._value_target(
                winner_side=game.winner_side,
                player_side=move.player_side,
            )
            samples.append(
                TrainingSample(
                    game_id=game_id,
                    move_id=move.id,
                    ply=move.ply,
                    player_side=move.player_side,
                    observation=move.board_before,
                    policy_target={str(move.action_idx): 1.0},
                    value_target=value_target,
                    split=split,
                    source=source,
                )
            )
        return await self.repository.bulk_create(samples)

    @staticmethod
    def _value_target(*, winner_side: WinnerSide, player_side: PlayerSide) -> float:
        if winner_side == WinnerSide.DRAW:
            return 0.0
        if winner_side == WinnerSide.P1:
            return 1.0 if player_side == PlayerSide.P1 else -1.0
        return 1.0 if player_side == PlayerSide.P2 else -1.0
