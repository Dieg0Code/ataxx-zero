import { apiDeleteNoContent, apiGet, apiPost } from "@/shared/api/client";
import type { BoardState, Move } from "@/features/match/types";
import type { MoveMode } from "@/features/match/api";

type StoredMoveResponse = {
  r1: number | null;
  c1: number | null;
  r2: number | null;
  c2: number | null;
  ply: number;
  board_after: BoardState | null;
};

type CreateGameResponse = {
  id: string;
};

type ActiveSeasonResponse = {
  id: string;
};

type BotProfileResponse = {
  user_id: string;
  agent_type: "heuristic" | "model";
  heuristic_level: "easy" | "normal" | "hard" | null;
  model_mode: "fast" | "strong" | null;
  enabled: boolean;
};

type BotProfileListResponse = {
  items: BotProfileResponse[];
};

export type PersistedReplay = {
  game: {
    id: string;
    season_id?: string | null;
    queue_type?: string;
    status: string;
    rated?: boolean;
    player1_id?: string | null;
    player2_id?: string | null;
    player1_username?: string | null;
    player2_username?: string | null;
    player1_agent?: string;
    player2_agent?: string;
    model_version_id?: string | null;
    winner_side: string | null;
    winner_user_id: string | null;
    termination_reason: string | null;
    source?: string;
    quality_score?: number | null;
    is_training_eligible?: boolean;
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

export type PersistedGameSummary = {
  id: string;
  queue_type: string;
  status: string;
  rated: boolean;
  player1_id: string | null;
  player2_id: string | null;
  player1_agent: "human" | "heuristic" | "model";
  player2_agent: "human" | "heuristic" | "model";
};

export type PersistedGameWsEvent =
  | {
      type: "game.subscribed";
      game_id: string;
      status: string;
    }
  | {
      type: "game.move.applied";
      game_id: string;
      move: StoredMoveResponse;
      game: PersistedReplay["game"];
    };

type PersistedOpponentAgent = "model" | "heuristic";
export type PersistedMoveMode = MoveMode | "manual";
type HeuristicLevel = "easy" | "normal" | "hard";

export type CreatePersistedGameOptions = {
  ranked?: boolean;
  preferredHeuristicLevel?: HeuristicLevel;
  preferredModelMode?: "fast" | "strong";
  selectedP1BotUserId?: string;
  selectedP2BotUserId?: string;
  player1Agent?: "human" | PersistedOpponentAgent;
  player2Agent?: PersistedOpponentAgent;
};

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export async function createPersistedGame(
  token: string,
  opponentAgent: PersistedOpponentAgent,
  options?: CreatePersistedGameOptions,
): Promise<string> {
  const ranked = options?.ranked ?? false;
  const requestedPlayer1Agent = options?.player1Agent ?? "human";
  const requestedPlayer2Agent = options?.player2Agent ?? opponentAgent;
  if (ranked && requestedPlayer1Agent !== "human") {
    throw new Error("IA vs IA no puede jugar ranked. Usa modo casual.");
  }
  if (ranked) {
    const season = await apiGet<ActiveSeasonResponse>("/api/v1/ranking/seasons/active", {
      headers: authHeaders(token),
    });
    const botsPage = await apiGet<BotProfileListResponse>("/api/v1/identity/bots?limit=200&offset=0", {
      headers: authHeaders(token),
    });
    const enabledBots = botsPage.items.filter((bot) => bot.enabled);
    const candidateBots = enabledBots.filter((bot) => bot.agent_type === requestedPlayer2Agent);
    if (candidateBots.length === 0) {
      throw new Error(`No hay bots disponibles para agente ${requestedPlayer2Agent}.`);
    }
    const selectedP2Bot =
      typeof options?.selectedP2BotUserId === "string" && options.selectedP2BotUserId.length > 0
        ? candidateBots.find((bot) => bot.user_id === options.selectedP2BotUserId)
        : undefined;
    const preferredBot =
      requestedPlayer2Agent === "heuristic"
        ? candidateBots.find((bot) => bot.heuristic_level === (options?.preferredHeuristicLevel ?? "normal"))
        : candidateBots.find((bot) => bot.model_mode === (options?.preferredModelMode ?? "fast"));
    const p2Bot = selectedP2Bot ?? preferredBot ?? candidateBots[0];
    const p1Bot =
      requestedPlayer1Agent === "human"
        ? undefined
        : enabledBots.find((bot) => bot.user_id === options?.selectedP1BotUserId);
    if (requestedPlayer1Agent !== "human" && p1Bot === undefined) {
      throw new Error("Para IA vs IA ranked debes seleccionar una cuenta bot valida para P1.");
    }
    const response = await apiPost<
      CreateGameResponse,
      {
        season_id: string;
        queue_type: "ranked";
        status: "in_progress";
        rated: true;
        player1_id?: string;
        player2_id: string;
        player1_agent: "human" | PersistedOpponentAgent;
        player2_agent: PersistedOpponentAgent;
        source: "human";
        is_training_eligible: false;
      }
    >(
      "/api/v1/gameplay/games",
      {
        season_id: season.id,
        queue_type: "ranked",
        status: "in_progress",
        rated: true,
        player1_id: p1Bot?.user_id,
        player2_id: p2Bot.user_id,
        player1_agent: requestedPlayer1Agent,
        player2_agent: requestedPlayer2Agent,
        source: "human",
        is_training_eligible: false,
      },
      { headers: authHeaders(token) },
    );
    return response.id;
  }

  const response = await apiPost<
    CreateGameResponse,
    {
      queue_type: "vs_ai";
      status: "in_progress";
      rated: false;
      player1_id?: string;
      player2_id?: string;
      player1_agent: "human" | PersistedOpponentAgent;
      player2_agent: PersistedOpponentAgent;
      source: "human";
      is_training_eligible: false;
    }
  >(
    "/api/v1/gameplay/games",
    {
      queue_type: "vs_ai",
      status: "in_progress",
      rated: false,
      player1_id: options?.selectedP1BotUserId,
      player2_id: options?.selectedP2BotUserId,
      player1_agent: requestedPlayer1Agent,
      player2_agent: requestedPlayer2Agent,
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
  mode: PersistedMoveMode = "manual",
): Promise<{ boardAfter: BoardState | null; move: Move | null }> {
  const response = await apiPost<StoredMoveResponse, { board: BoardState; move: Move; mode: PersistedMoveMode }>(
    `/api/v1/gameplay/games/${gameId}/move/manual`,
    { board, move, mode },
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

export async function fetchPersistedGameSummary(
  token: string,
  gameId: string,
): Promise<PersistedGameSummary> {
  return apiGet<PersistedGameSummary>(`/api/v1/gameplay/games/${gameId}`, {
    headers: authHeaders(token),
  });
}

export async function deletePersistedGame(token: string, gameId: string): Promise<void> {
  await apiDeleteNoContent(`/api/v1/gameplay/games/${gameId}`, {
    headers: authHeaders(token),
  });
}

export function openPersistedGameSocket(
  token: string,
  gameId: string,
  onEvent: (event: PersistedGameWsEvent) => void,
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/api/v1/gameplay/games/${gameId}/ws?token=${encodeURIComponent(
    token,
  )}`;
  const socket = new WebSocket(wsUrl);
  socket.onmessage = (messageEvent) => {
    try {
      const parsed = JSON.parse(messageEvent.data) as PersistedGameWsEvent;
      onEvent(parsed);
    } catch {
      // Ignore malformed payloads.
    }
  };
  return socket;
}
