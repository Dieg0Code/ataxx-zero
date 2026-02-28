import { apiGet, apiPost } from "@/shared/api/client";

export type QueueStatus = "idle" | "waiting" | "matched" | "canceled";
export type MatchedWith = "human" | "bot";

type QueueApiResponse = {
  queue_id: string | null;
  status: QueueStatus;
  season_id: string | null;
  game_id: string | null;
  matched_with: MatchedWith | null;
  created_at: string | null;
  updated_at: string | null;
};

type QueueLeaveApiResponse = {
  left_queue: boolean;
  status: "idle" | "canceled";
};

type QueueDecisionApiResponse = {
  decision: "accepted" | "rejected";
  queue_id: string;
  status: "waiting" | "matched" | "canceled";
  game_id: string | null;
  updated_at: string;
};

export type QueueWsEvent =
  | {
      type: "queue.subscribed";
    }
  | {
      type: "queue.status";
      payload: QueueApiResponse;
    };

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export async function joinRankedQueue(token: string): Promise<QueueApiResponse> {
  return apiPost<QueueApiResponse, Record<string, never>>(
    "/api/v1/matchmaking/queue/join",
    {},
    { headers: authHeaders(token) },
  );
}

export async function fetchQueueStatus(token: string): Promise<QueueApiResponse> {
  return apiGet<QueueApiResponse>("/api/v1/matchmaking/queue/status", {
    headers: authHeaders(token),
  });
}

export async function leaveQueue(token: string): Promise<QueueLeaveApiResponse> {
  return apiPost<QueueLeaveApiResponse, Record<string, never>>(
    "/api/v1/matchmaking/queue/leave",
    {},
    { headers: authHeaders(token) },
  );
}

export async function acceptMatchedQueue(token: string): Promise<QueueDecisionApiResponse> {
  return apiPost<QueueDecisionApiResponse, Record<string, never>>(
    "/api/v1/matchmaking/queue/accept",
    {},
    { headers: authHeaders(token) },
  );
}

export async function rejectMatchedQueue(token: string): Promise<QueueDecisionApiResponse> {
  return apiPost<QueueDecisionApiResponse, Record<string, never>>(
    "/api/v1/matchmaking/queue/reject",
    {},
    { headers: authHeaders(token) },
  );
}

export function openQueueSocket(
  token: string,
  onEvent: (event: QueueWsEvent) => void,
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/api/v1/matchmaking/queue/ws?token=${encodeURIComponent(
    token,
  )}`;
  const socket = new WebSocket(wsUrl);
  socket.onmessage = (messageEvent) => {
    try {
      const parsed = JSON.parse(messageEvent.data) as QueueWsEvent;
      onEvent(parsed);
    } catch {
      // Ignore malformed payloads.
    }
  };
  return socket;
}
