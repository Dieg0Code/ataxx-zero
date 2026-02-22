import { apiGet, apiPost } from "@/shared/api/client";
import type { BoardState, Move } from "@/features/match/types";

type StoredMoveResponse = {
  r1: number | null;
  c1: number | null;
  r2: number | null;
  c2: number | null;
  board_after: BoardState | null;
};

type CreateGameResponse = {
  id: string;
};

export type PersistedReplay = {
  game: {
    id: string;
    status: string;
    winner_side: string | null;
    winner_user_id: string | null;
    termination_reason: string | null;
  };
  moves: Array<{
    ply: number;
    player_side: "p1" | "p2";
    r1: number | null;
    c1: number | null;
    r2: number | null;
    c2: number | null;
    board_before: BoardState | null;
    board_after: BoardState | null;
    mode: string;
  }>;
};

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export async function createPersistedGame(token: string): Promise<string> {
  const response = await apiPost<
    CreateGameResponse,
    {
      queue_type: "vs_ai";
      rated: false;
      player1_agent: "human";
      player2_agent: "model";
      source: "human";
      is_training_eligible: false;
    }
  >(
    "/api/v1/gameplay/games",
    {
      queue_type: "vs_ai",
      rated: false,
      player1_agent: "human",
      player2_agent: "model",
      source: "human",
      is_training_eligible: false,
    },
    { headers: authHeaders(token) },
  );
  return response.id;
}

export async function storeManualMove(
  token: string,
  gameId: string,
  board: BoardState,
  move: Move,
): Promise<{ boardAfter: BoardState | null; move: Move | null }> {
  const response = await apiPost<StoredMoveResponse, { board: BoardState; move: Move }>(
    `/api/v1/gameplay/games/${gameId}/move/manual`,
    { board, move },
    { headers: authHeaders(token) },
  );
  const persistedMove =
    response.r1 === null || response.c1 === null || response.r2 === null || response.c2 === null
      ? null
      : { r1: response.r1, c1: response.c1, r2: response.r2, c2: response.c2 };
  return { boardAfter: response.board_after, move: persistedMove };
}

export async function storeInferredMove(
  token: string,
  gameId: string,
  board: BoardState,
  mode: "fast" | "strong",
): Promise<{ boardAfter: BoardState | null; move: Move | null }> {
  const response = await apiPost<StoredMoveResponse, { board: BoardState; mode: "fast" | "strong" }>(
    `/api/v1/gameplay/games/${gameId}/move`,
    { board, mode },
    { headers: authHeaders(token) },
  );
  const persistedMove =
    response.r1 === null || response.c1 === null || response.r2 === null || response.c2 === null
      ? null
      : { r1: response.r1, c1: response.c1, r2: response.r2, c2: response.c2 };
  return { boardAfter: response.board_after, move: persistedMove };
}

export async function fetchPersistedReplay(token: string, gameId: string): Promise<PersistedReplay> {
  return apiGet<PersistedReplay>(`/api/v1/gameplay/games/${gameId}/replay`, {
    headers: authHeaders(token),
  });
}
