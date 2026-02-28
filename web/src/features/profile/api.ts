import { apiDeleteNoContent, apiGet } from "@/shared/api/client";

export type ProfileGame = {
  id: string;
  season_id: string | null;
  queue_type: string;
  status: string;
  rated: boolean;
  player1_id: string | null;
  player2_id: string | null;
  player1_agent: string;
  player2_agent: string;
  model_version_id: string | null;
  winner_side: string | null;
  winner_user_id: string | null;
  termination_reason: string | null;
  source: string;
  quality_score: number | null;
  is_training_eligible: boolean;
};

export type ProfileGamePage = {
  items: ProfileGame[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

export async function fetchMyGames(
  accessToken: string,
  limit = 20,
  offset = 0,
  statuses?: string[],
): Promise<ProfileGamePage> {
  const statusQuery =
    statuses && statuses.length > 0
      ? `&${statuses.map((status) => `status=${encodeURIComponent(status)}`).join("&")}`
      : "";
  return apiGet<ProfileGamePage>(`/api/v1/gameplay/games?limit=${limit}&offset=${offset}${statusQuery}`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
}

export async function deleteMyGame(accessToken: string, gameId: string): Promise<void> {
  return apiDeleteNoContent(`/api/v1/gameplay/games/${gameId}`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
}
