import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/widgets/layout/AppShell";
import { renderWithProviders } from "@/test/render";

const logoutMock = vi.fn();
const openInvitationsSocketMock = vi.fn();
const acceptInvitationMock = vi.fn();
const rejectInvitationMock = vi.fn();
const fetchActiveSeasonMock = vi.fn();
const fetchLeaderboardMock = vi.fn();
const fetchUserRatingMock = vi.fn();
const fetchRatingEventsMock = vi.fn();
const fetchMyGamesMock = vi.fn();
let authState: { isAuthenticated: boolean; user: { id: string; username: string } | null } = {
  isAuthenticated: true,
  user: { id: "user-1", username: "test" },
};

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => ({
    ...authState,
    loading: false,
    accessToken: authState.isAuthenticated ? "token" : null,
    register: vi.fn(),
    login: vi.fn(),
    logout: logoutMock,
    refreshUser: vi.fn(),
  }),
}));

vi.mock("@/features/matches/api", () => ({
  openInvitationsSocket: (...args: unknown[]) => openInvitationsSocketMock(...args),
  acceptInvitation: (...args: unknown[]) => acceptInvitationMock(...args),
  rejectInvitation: (...args: unknown[]) => rejectInvitationMock(...args),
}));

vi.mock("@/features/ranking/api", () => ({
  fetchActiveSeason: (...args: unknown[]) => fetchActiveSeasonMock(...args),
  fetchLeaderboard: (...args: unknown[]) => fetchLeaderboardMock(...args),
  fetchUserRating: (...args: unknown[]) => fetchUserRatingMock(...args),
  fetchRatingEvents: (...args: unknown[]) => fetchRatingEventsMock(...args),
}));

vi.mock("@/features/profile/api", () => ({
  fetchMyGames: (...args: unknown[]) => fetchMyGamesMock(...args),
}));

describe("AppShell", () => {
  beforeEach(() => {
    logoutMock.mockReset();
    openInvitationsSocketMock.mockReset();
    acceptInvitationMock.mockReset();
    rejectInvitationMock.mockReset();
    fetchActiveSeasonMock.mockReset();
    fetchLeaderboardMock.mockReset();
    fetchUserRatingMock.mockReset();
    fetchRatingEventsMock.mockReset();
    fetchMyGamesMock.mockReset();
    authState = {
      isAuthenticated: true,
      user: { id: "user-1", username: "test" },
    };
    openInvitationsSocketMock.mockReturnValue({
      close: vi.fn(),
      onerror: null,
      onmessage: null,
      onclose: null,
    });
    acceptInvitationMock.mockResolvedValue({});
    rejectInvitationMock.mockResolvedValue({});
    fetchActiveSeasonMock.mockResolvedValue({
      id: "season-1",
      name: "S1",
      starts_at: "2026-01-01T00:00:00.000Z",
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
    fetchUserRatingMock.mockResolvedValue({
      id: "rating-1",
      user_id: "user-1",
      season_id: "season-1",
      rating: 1200,
      games_played: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      updated_at: "2026-01-01T00:00:00.000Z",
      league: "Protocol",
      division: "III",
      lp: 0,
      next_major_promo: null,
    });
    fetchRatingEventsMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 6,
      offset: 0,
      has_more: false,
    });
    fetchMyGamesMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 8,
      offset: 0,
      has_more: false,
    });
  });

  it("renders nav with active item and authenticated action", () => {
    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: "/ranking" },
    );

    expect(screen.getByText("underbyteLabs")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Ranking" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Salir (test)" })).toBeInTheDocument();
    expect(screen.getByText("contenido")).toBeInTheDocument();
  });

  it("marks profile tab active on profile subroutes", () => {
    renderWithProviders(
      <AppShell>
        <div>detalle</div>
      </AppShell>,
      { route: "/profile/games/game-123" },
    );

    expect(screen.getByRole("link", { name: "Perfil" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Inicio" })).not.toHaveAttribute("aria-current");
  });

  it("renders login action when user is not authenticated", () => {
    authState = {
      isAuthenticated: false,
      user: null,
    };

    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: "/" },
    );

    expect(screen.getByRole("link", { name: "Entrar" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Salir/ })).not.toBeInTheDocument();
  });

  it("shows and consumes flash message from navigation state", () => {
    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: { pathname: "/", state: { flash: "Sesion iniciada correctamente." } } },
    );

    expect(screen.getByText("Sesion iniciada correctamente.")).toBeInTheDocument();
  });

  it("renders warning flash variant", () => {
    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: { pathname: "/", state: { flash: { message: "Cola pausada.", tone: "warning" } } } },
    );

    expect(screen.getByText("Cola pausada.")).toBeInTheDocument();
  });

  it("shows pending invitations badge on profile nav item", async () => {
    let wsHandler: ((event: { type: string; payload?: { items: Array<{ status: string }> } }) => void) | null = null;
    openInvitationsSocketMock.mockImplementation((_token: string, onEvent: typeof wsHandler) => {
      wsHandler = onEvent;
      return {
        close: vi.fn(),
        onerror: null,
        onmessage: null,
        onclose: null,
      };
    });

    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: "/" },
    );

    await act(async () => {
      wsHandler?.({
        type: "invitations.status",
        payload: {
          items: [{ status: "pending" }, { status: "pending" }, { status: "aborted" }],
        },
      });
    });

    expect(await screen.findByLabelText("2 invitaciones pendientes")).toBeInTheDocument();
  });

  it("opens invitation panel and renders actions for a pending invitation", async () => {
    let wsHandler: ((event: { type: string; payload?: { items: Array<{ id: string; status: string }> } }) => void) | null = null;
    openInvitationsSocketMock.mockImplementation((_token: string, onEvent: typeof wsHandler) => {
      wsHandler = onEvent;
      return {
        close: vi.fn(),
        onerror: null,
        onmessage: null,
        onclose: null,
      };
    });

    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: "/" },
    );

    await act(async () => {
      wsHandler?.({
        type: "invitations.status",
        payload: {
          items: [{ id: "invite-abc12345", status: "pending" }],
        },
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "Ver invitaciones" }));
    expect(await screen.findByText("invite-a")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Aceptar" })).toBeInTheDocument();
  });

  it("prefetches ranking and profile data on nav hover", async () => {
    renderWithProviders(
      <AppShell>
        <div>contenido</div>
      </AppShell>,
      { route: "/" },
    );

    fireEvent.mouseEnter(screen.getByRole("link", { name: "Ranking" }));
    await waitFor(() => {
      expect(fetchActiveSeasonMock).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(fetchLeaderboardMock).toHaveBeenCalledWith("season-1", 10, 0, {
        competitorFilter: "all",
        query: "",
      });
    });
    await waitFor(() => {
      expect(fetchUserRatingMock).toHaveBeenCalledWith("user-1", "season-1");
    });

    fireEvent.mouseEnter(screen.getByRole("link", { name: "Perfil" }));
    await waitFor(() => {
      expect(fetchMyGamesMock).toHaveBeenCalledWith("token", 8, 0, ["finished"]);
    });
    await waitFor(() => {
      expect(fetchRatingEventsMock).toHaveBeenCalledWith("user-1", "season-1", 6, 0);
    });
  });

});
