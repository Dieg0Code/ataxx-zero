import { apiGet } from "@/shared/api/client";

export type PlayableBot = {
  user_id: string;
  username: string;
  bot_kind: "heuristic" | "model" | null;
  agent_type: "heuristic" | "model";
  heuristic_level: "easy" | "normal" | "hard" | null;
  model_mode: "fast" | "strong" | null;
  enabled: boolean;
};

export type PublicPlayer = {
  user_id: string;
  username: string;
  is_bot: boolean;
  bot_kind: "heuristic" | "model" | null;
  agent_type: "heuristic" | "model" | null;
  heuristic_level: "easy" | "normal" | "hard" | null;
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
