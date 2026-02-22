import { apiGet } from "@/shared/api/client";

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

export async function fetchMyGames(accessToken: string, limit = 20, offset = 0): Promise<ProfileGamePage> {
  return apiGet<ProfileGamePage>(`/api/v1/gameplay/games?limit=${limit}&offset=${offset}`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
}

