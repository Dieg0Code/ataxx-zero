import { apiGet } from "@/shared/api/client";

export type SeasonResponse = {
  id: string;
  name: string;
  starts_at: string;
  ends_at: string | null;
  is_active: boolean;
};

export type LeaderboardEntry = {
  season_id: string;
  user_id: string;
  rank: number;
  rating: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  computed_at: string;
};

export type Paged<T> = {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

export async function fetchActiveSeason(): Promise<SeasonResponse> {
  return apiGet<SeasonResponse>("/api/v1/ranking/seasons/active");
}

export async function fetchLeaderboard(
  seasonId: string,
  limit: number,
  offset: number
): Promise<Paged<LeaderboardEntry>> {
  return apiGet<Paged<LeaderboardEntry>>(
    `/api/v1/ranking/leaderboard/${seasonId}?limit=${limit}&offset=${offset}`
  );
}
