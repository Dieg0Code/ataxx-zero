import { apiGet } from "@/shared/api/client";

export type PlayableBot = {
  user_id: string;
  username: string;
  bot_kind: "heuristic" | "model" | null;
  agent_type: "heuristic" | "model";
  heuristic_level: "easy" | "normal" | "hard" | "apex" | "gambit" | "sentinel" | null;
  model_mode: "fast" | "strong" | null;
  enabled: boolean;
};

export type PublicPlayer = {
  user_id: string;
  username: string;
  is_bot: boolean;
  bot_kind: "heuristic" | "model" | null;
  agent_type: "heuristic" | "model" | null;
  heuristic_level: "easy" | "normal" | "hard" | "apex" | "gambit" | "sentinel" | null;
  model_mode: "fast" | "strong" | null;
  enabled: boolean | null;
};

type PlayableBotPage = {
  items: PlayableBot[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

type PublicPlayerPage = {
  items: PublicPlayer[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

const PUBLIC_PLAYERS_PREFETCH_KEY = "ataxx.identity.players.prefetch.v1";
const PUBLIC_PLAYERS_PREFETCH_TTL_MS = 60_000;

type PublicPlayersPrefetchSnapshot = {
  created_at: number;
  items: PublicPlayer[];
};

export async function fetchPlayableBots(accessToken: string, limit = 200): Promise<PlayableBot[]> {
  const page = await apiGet<PlayableBotPage>(`/api/v1/identity/bots?limit=${limit}&offset=0`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  return page.items.filter((bot) => bot.enabled);
}

export async function fetchPublicPlayers(
  accessToken: string,
  { limit = 200, query = "" }: { limit?: number; query?: string } = {},
): Promise<PublicPlayer[]> {
  const q = query.trim();
  const qParam = q.length > 0 ? `&q=${encodeURIComponent(q)}` : "";
  const page = await apiGet<PublicPlayerPage>(
    `/api/v1/identity/players?limit=${limit}&offset=0${qParam}`,
    {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    },
  );
  return page.items;
}

function writePrefetchedPublicPlayers(items: PublicPlayer[]): void {
  try {
    const snapshot: PublicPlayersPrefetchSnapshot = {
      created_at: Date.now(),
      items,
    };
    sessionStorage.setItem(PUBLIC_PLAYERS_PREFETCH_KEY, JSON.stringify(snapshot));
  } catch {
    // Ignore storage failures (private mode/quota) and continue without prefetch cache.
  }
}

export function readPrefetchedPublicPlayers(maxAgeMs = PUBLIC_PLAYERS_PREFETCH_TTL_MS): PublicPlayer[] | null {
  try {
    const raw = sessionStorage.getItem(PUBLIC_PLAYERS_PREFETCH_KEY);
    if (raw === null) {
      return null;
    }
    const parsed = JSON.parse(raw) as PublicPlayersPrefetchSnapshot;
    if (!Array.isArray(parsed.items) || typeof parsed.created_at !== "number") {
      sessionStorage.removeItem(PUBLIC_PLAYERS_PREFETCH_KEY);
      return null;
    }
    if (Date.now() - parsed.created_at > maxAgeMs) {
      sessionStorage.removeItem(PUBLIC_PLAYERS_PREFETCH_KEY);
      return null;
    }
    return parsed.items;
  } catch {
    return null;
  }
}

export async function prefetchPublicPlayers(
  accessToken: string,
  { limit = 200, query = "", maxAgeMs = PUBLIC_PLAYERS_PREFETCH_TTL_MS }: { limit?: number; query?: string; maxAgeMs?: number } = {},
): Promise<PublicPlayer[]> {
  const trimmedQuery = query.trim();
  const canUseCache = trimmedQuery.length === 0;

  if (canUseCache) {
    const cached = readPrefetchedPublicPlayers(maxAgeMs);
    if (cached !== null) {
      return cached;
    }
  }

  const players = await fetchPublicPlayers(accessToken, { limit, query });
  if (canUseCache) {
    writePrefetchedPublicPlayers(players);
  }
  return players;
}
