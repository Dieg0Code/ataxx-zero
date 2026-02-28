import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RankingPage } from "@/pages/ranking/RankingPage";
import { renderWithProviders } from "@/test/render";

const fetchActiveSeasonMock = vi.fn();
const fetchLeaderboardMock = vi.fn();
const fetchUserRatingMock = vi.fn();
const useAuthMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/features/ranking/api", () => ({
  fetchActiveSeason: (...args: unknown[]) => fetchActiveSeasonMock(...args),
  fetchLeaderboard: (...args: unknown[]) => fetchLeaderboardMock(...args),
  fetchUserRating: (...args: unknown[]) => fetchUserRatingMock(...args),
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => useAuthMock(),
}));

describe("RankingPage", () => {
  beforeEach(() => {
    fetchActiveSeasonMock.mockReset();
    fetchLeaderboardMock.mockReset();
    fetchUserRatingMock.mockReset();
    useAuthMock.mockReturnValue({
      user: null,
      loading: false,
      isAuthenticated: false,
      accessToken: null,
      register: vi.fn(),
      login: vi.fn(),
      logout: vi.fn(),
      refreshUser: vi.fn(),
    });
  });

  it("renders username instead of raw id when available", async () => {
    fetchActiveSeasonMock.mockResolvedValue({
      id: "season-1",
      name: "Season 1",
      starts_at: "2026-02-22T00:00:00Z",
      ends_at: null,
      is_active: true,
    });
    fetchLeaderboardMock.mockResolvedValue({
      items: [
        {
          season_id: "season-1",
          user_id: "11111111-1111-1111-1111-111111111111",
          username: "dieg0",
          rank: 1,
          rating: 1242.5,
          wins: 6,
          losses: 2,
          draws: 1,
          win_rate: 0.66,
          computed_at: "2026-02-22T00:00:00Z",
        },
      ],
      total: 1,
      limit: 10,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<RankingPage />, { route: "/ranking" });

    await waitFor(() => {
      expect(fetchLeaderboardMock).toHaveBeenCalledWith("season-1", 10, 0, {
        competitorFilter: "all",
        query: "",
      });
    });

    expect(await screen.findByText("dieg0")).toBeInTheDocument();
    expect(screen.queryByText("11111111")).not.toBeInTheDocument();
  });

  it("shows competitive summary card for authenticated user", async () => {
    useAuthMock.mockReturnValue({
      user: { id: "user-1", username: "test-user" },
      loading: false,
      isAuthenticated: true,
      accessToken: "token",
      register: vi.fn(),
      login: vi.fn(),
      logout: vi.fn(),
      refreshUser: vi.fn(),
    });

    fetchActiveSeasonMock.mockResolvedValue({
      id: "season-1",
      name: "Season 1",
      starts_at: "2026-02-22T00:00:00Z",
      ends_at: null,
      is_active: true,
    });
    fetchUserRatingMock.mockResolvedValue({
      id: "rating-1",
      user_id: "user-1",
      season_id: "season-1",
      rating: 1300.2,
      games_played: 10,
      wins: 7,
      losses: 2,
      draws: 1,
      updated_at: "2026-02-22T00:00:00Z",
      league: "Root Access",
      division: "II",
      lp: 78,
      next_major_promo: "Kernel Breach",
    });
    fetchLeaderboardMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 10,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<RankingPage />, { route: "/ranking" });

    expect(await screen.findByText("Tu posicion")).toBeInTheDocument();
    expect(await screen.findByText("Root Access II")).toBeInTheDocument();
    expect(await screen.findByText("78 LP")).toBeInTheDocument();
  });

  it("applies bot filter on leaderboard request", async () => {
    fetchActiveSeasonMock.mockResolvedValue({
      id: "season-1",
      name: "Season 1",
      starts_at: "2026-02-22T00:00:00Z",
      ends_at: null,
      is_active: true,
    });
    fetchLeaderboardMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 10,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<RankingPage />, { route: "/ranking" });
    expect(await screen.findByText("Global")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Bots" }));

    await waitFor(() => {
      expect(fetchLeaderboardMock).toHaveBeenLastCalledWith("season-1", 10, 0, {
        competitorFilter: "bots",
        query: "",
      });
    });
  });
});
