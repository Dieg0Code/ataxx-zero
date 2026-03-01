import { apiGet, apiPost } from "@/shared/api/client";
import { buildWsUrl } from "@/shared/api/ws";

export type HumanInvitation = {
  id: string;
  queue_type: "custom";
  status: "pending" | "in_progress" | "aborted" | "finished";
  rated?: boolean;
  player1_id: string | null;
  player2_id: string | null;
  created_by_user_id: string | null;
  player1_agent: "human";
  player2_agent: "human";
  created_at?: string;
};

type PagedInvitations = {
  items: HumanInvitation[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

type InvitationsWsPayload = {
  total: number;
  items: HumanInvitation[];
};

export type InvitationsWsEvent =
  | {
      type: "invitations.subscribed";
    }
  | {
      type: "invitations.status";
      payload: InvitationsWsPayload;
    };

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export async function createHumanInvitation(
  token: string,
  opponentUserId: string,
): Promise<HumanInvitation> {
  return apiPost<HumanInvitation, { opponent_user_id: string }>(
    "/api/v1/matches/invitations",
    { opponent_user_id: opponentUserId },
    { headers: authHeaders(token) },
  );
}

export async function fetchIncomingInvitations(
  token: string,
  limit = 10,
  offset = 0,
): Promise<PagedInvitations> {
  return apiGet<PagedInvitations>(
    `/api/v1/matches/invitations/incoming?limit=${limit}&offset=${offset}`,
    { headers: authHeaders(token) },
  );
}

export async function fetchInvitationGame(
  token: string,
  gameId: string,
): Promise<HumanInvitation> {
  return apiGet<HumanInvitation>(`/api/v1/matches/${gameId}`, {
    headers: authHeaders(token),
  });
}

export async function acceptInvitation(
  token: string,
  gameId: string,
): Promise<HumanInvitation> {
  return apiPost<HumanInvitation, Record<string, never>>(
    `/api/v1/matches/invitations/${gameId}/accept`,
    {},
    { headers: authHeaders(token) },
  );
}

export async function rejectInvitation(
  token: string,
  gameId: string,
): Promise<HumanInvitation> {
  return apiPost<HumanInvitation, Record<string, never>>(
    `/api/v1/matches/invitations/${gameId}/reject`,
    {},
    { headers: authHeaders(token) },
  );
}

export function openInvitationsSocket(
  token: string,
  onEvent: (event: InvitationsWsEvent) => void,
): WebSocket {
  const wsUrl = buildWsUrl("/api/v1/matches/invitations/ws", { token });
  const socket = new WebSocket(wsUrl);
  socket.onmessage = (messageEvent) => {
    try {
      const parsed = JSON.parse(messageEvent.data) as InvitationsWsEvent;
      onEvent(parsed);
    } catch {
      // Ignore malformed payloads to keep socket alive.
    }
  };
  return socket;
}
