import { apiPost } from "@/shared/api/client";
import type { BoardState, Move } from "@/features/match/types";

export type MoveMode =
  | "fast"
  | "strong"
  | "heuristic_easy"
  | "heuristic_normal"
  | "heuristic_hard"
  | "random";

type MovePayload = {
  r1: number;
  c1: number;
  r2: number;
  c2: number;
};

type PredictMoveResponse = {
  move: MovePayload | null;
  action_idx: number;
  value: number;
  mode: MoveMode;
};

export async function predictAIMove(
  board: BoardState,
  mode: MoveMode,
): Promise<{ move: Move | null; value: number }> {
  const response = await apiPost<PredictMoveResponse, { board: BoardState; mode: MoveMode }>(
    "/api/v1/gameplay/move",
    { board, mode },
  );
  if (response.move === null) {
    return { move: null, value: response.value };
  }
  return {
    move: {
      r1: response.move.r1,
      c1: response.move.c1,
      r2: response.move.r2,
      c2: response.move.c2,
    },
    value: response.value,
  };
}
