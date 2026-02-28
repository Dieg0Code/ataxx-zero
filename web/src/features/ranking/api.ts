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
  username?: string | null;
  is_bot?: boolean;
  bot_kind?: string | null;
  rank: number;
  rating: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  computed_at: string;
  league?: string;
  division?: string;
  lp?: number;
  recent_lp_delta?: number | null;
  recent_transition_type?: string | null;
  next_major_promo?: string | null;
  prestige_title?: string | null;
};

export type RatingResponse = {
  id: string;
  user_id: string;
  season_id: string;
  rating: number;
  games_played: number;
  wins: number;
  losses: number;
  draws: number;
  updated_at: string;
  league: string;
  division: string;
  lp: number;
  next_major_promo: string | null;
};

export type RatingEventResponse = {
  id: string;
  game_id: string;
  user_id: string;
  season_id: string;
  rating_before: number;
  rating_after: number;
  delta: number;
  before_league: string | null;
  before_division: string | null;
  before_lp: number | null;
  after_league: string | null;
  after_division: string | null;
  after_lp: number | null;
  transition_type: "promotion" | "demotion" | "stable";
  major_promo_name: string | null;
  created_at: string;
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
  offset: number,
  options?: {
    competitorFilter?: "all" | "humans" | "bots";
    query?: string;
  }
): Promise<Paged<LeaderboardEntry>> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (options?.competitorFilter && options.competitorFilter !== "all") {
    params.set("competitor_filter", options.competitorFilter);
  }
  if (options?.query && options.query.trim().length > 0) {
    params.set("q", options.query.trim());
  }
  return apiGet<Paged<LeaderboardEntry>>(
    `/api/v1/ranking/leaderboard/${seasonId}?${params.toString()}`
  );
}

export async function fetchUserRating(userId: string, seasonId: string): Promise<RatingResponse> {
  return apiGet<RatingResponse>(`/api/v1/ranking/ratings/${userId}/${seasonId}`);
}

export async function fetchRatingEvents(
  userId: string,
  seasonId: string,
  limit = 8,
  offset = 0
): Promise<Paged<RatingEventResponse>> {
  return apiGet<Paged<RatingEventResponse>>(
    `/api/v1/ranking/ratings/${userId}/${seasonId}/events?limit=${limit}&offset=${offset}`
  );
}
