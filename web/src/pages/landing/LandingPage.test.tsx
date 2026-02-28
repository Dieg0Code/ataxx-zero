import { fireEvent, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LandingPage } from "@/pages/landing/LandingPage";
import { renderWithProviders } from "@/test/render";

const fetchActiveSeasonMock = vi.fn();
const fetchLeaderboardMock = vi.fn();
const joinRankedQueueMock = vi.fn();
const openQueueSocketMock = vi.fn();
const leaveQueueMock = vi.fn();
const acceptMatchedQueueMock = vi.fn();
const rejectMatchedQueueMock = vi.fn();
const fetchPersistedGameSummaryMock = vi.fn();
const fetchPublicPlayersMock = vi.fn();
const useAuthMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/features/ranking/api", () => ({
  fetchActiveSeason: (...args: unknown[]) => fetchActiveSeasonMock(...args),
  fetchLeaderboard: (...args: unknown[]) => fetchLeaderboardMock(...args),
}));

vi.mock("@/features/matchmaking/api", () => ({
  joinRankedQueue: (...args: unknown[]) => joinRankedQueueMock(...args),
  openQueueSocket: (...args: unknown[]) => openQueueSocketMock(...args),
  leaveQueue: (...args: unknown[]) => leaveQueueMock(...args),
  acceptMatchedQueue: (...args: unknown[]) => acceptMatchedQueueMock(...args),
  rejectMatchedQueue: (...args: unknown[]) => rejectMatchedQueueMock(...args),
}));

vi.mock("@/features/match/persistence", () => ({
  fetchPersistedGameSummary: (...args: unknown[]) => fetchPersistedGameSummaryMock(...args),
}));

vi.mock("@/features/identity/api", () => ({
  fetchPublicPlayers: (...args: unknown[]) => fetchPublicPlayersMock(...args),
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => useAuthMock(),
}));

describe("LandingPage queue", () => {
  beforeEach(() => {
    fetchActiveSeasonMock.mockReset();
    fetchLeaderboardMock.mockReset();
    joinRankedQueueMock.mockReset();
    openQueueSocketMock.mockReset();
    leaveQueueMock.mockReset();
    acceptMatchedQueueMock.mockReset();
    rejectMatchedQueueMock.mockReset();
    fetchPersistedGameSummaryMock.mockReset();
    fetchPublicPlayersMock.mockReset();
    openQueueSocketMock.mockReturnValue({
      close: vi.fn(),
      onmessage: null,
      onerror: null,
      onclose: null,
    });
    useAuthMock.mockReturnValue({
      user: { id: "u1", username: "demo" },
      loading: false,
      isAuthenticated: true,
      accessToken: "token-123",
      register: vi.fn(),
      login: vi.fn(),
      logout: vi.fn(),
      refreshUser: vi.fn(),
    });
    fetchActiveSeasonMock.mockRejectedValue(new Error("no season"));
    fetchLeaderboardMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 3,
      offset: 0,
      has_more: false,
    });
    joinRankedQueueMock.mockResolvedValue({
      queue_id: "q-1",
      status: "waiting",
      season_id: "s-1",
      game_id: null,
      matched_with: null,
      created_at: "2026-02-23T00:00:00",
      updated_at: "2026-02-23T00:00:00",
    });
    acceptMatchedQueueMock.mockResolvedValue({
      decision: "accepted",
      queue_id: "q-1",
      status: "matched",
      game_id: "game-1",
      updated_at: "2026-02-23T00:00:01",
    });
    rejectMatchedQueueMock.mockResolvedValue({
      decision: "rejected",
      queue_id: "q-1",
      status: "canceled",
      game_id: null,
      updated_at: "2026-02-23T00:00:01",
    });
    fetchPersistedGameSummaryMock.mockResolvedValue({
      id: "game-1",
      queue_type: "ranked",
      status: "matched",
      rated: true,
      player1_id: "u1",
      player2_id: "bot-1",
      player1_agent: "human",
      player2_agent: "heuristic",
    });
    fetchPublicPlayersMock.mockResolvedValue([
      {
        user_id: "bot-1",
        username: "aetherglyph",
        is_bot: true,
        bot_kind: "heuristic",
        agent_type: "heuristic",
        heuristic_level: "easy",
        model_mode: null,
        enabled: true,
      },
    ]);
  });

  it("shows queue searching status after clicking Buscar partida", async () => {
    renderWithProviders(<LandingPage />, { route: "/" });

    fireEvent.click(screen.getByRole("button", { name: /buscar partida/i }));

    const searchingLabels = await screen.findAllByText(/buscando rival/i);
    expect(searchingLabels.length).toBeGreaterThan(0);
    expect(screen.getByText(/tiempo en cola/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /detener busqueda/i })).toBeInTheDocument();
  }, 15000);

  it("shows accept modal when queue already returns a matched game", async () => {
    joinRankedQueueMock.mockResolvedValue({
      queue_id: "q-2",
      status: "matched",
      season_id: "s-1",
      game_id: "game-1",
      matched_with: "bot",
      created_at: "2026-02-23T00:00:00",
      updated_at: "2026-02-23T00:00:01",
    });

    renderWithProviders(<LandingPage />, { route: "/" });
    fireEvent.click(screen.getByRole("button", { name: /buscar partida/i }));

    expect(await screen.findByText(/partida encontrada/i)).toBeInTheDocument();
    expect(screen.getByText(/rival:/i)).toBeInTheDocument();
    expect(screen.getByText(/aetherglyph/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /aceptar/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /rechazar/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cerrar modal/i })).toBeInTheDocument();
  }, 15000);

  it("opens guest modal when unauthenticated user clicks Buscar partida", async () => {
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

    renderWithProviders(<LandingPage />, { route: "/" });
    fireEvent.click(screen.getByRole("button", { name: /buscar partida/i }));

    expect(await screen.findByText(/modo invitado/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /entrar como invitado/i })).toBeInTheDocument();
  });

});
