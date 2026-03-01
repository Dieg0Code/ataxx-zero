import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProfilePage } from "@/pages/profile/ProfilePage";
import { renderWithProviders } from "@/test/render";

const fetchMyGamesMock = vi.fn();
const deleteMyGameMock = vi.fn();
const fetchActiveSeasonMock = vi.fn();
const fetchUserRatingMock = vi.fn();
const fetchRatingEventsMock = vi.fn();
const fetchIncomingInvitationsMock = vi.fn();
const acceptInvitationMock = vi.fn();
const rejectInvitationMock = vi.fn();
const openInvitationsSocketMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/features/profile/api", () => ({
  fetchMyGames: (...args: unknown[]) => fetchMyGamesMock(...args),
  deleteMyGame: (...args: unknown[]) => deleteMyGameMock(...args),
}));

vi.mock("@/features/ranking/api", () => ({
  fetchActiveSeason: (...args: unknown[]) => fetchActiveSeasonMock(...args),
  fetchUserRating: (...args: unknown[]) => fetchUserRatingMock(...args),
  fetchRatingEvents: (...args: unknown[]) => fetchRatingEventsMock(...args),
}));
vi.mock("@/features/matches/api", () => ({
  fetchIncomingInvitations: (...args: unknown[]) => fetchIncomingInvitationsMock(...args),
  acceptInvitation: (...args: unknown[]) => acceptInvitationMock(...args),
  rejectInvitation: (...args: unknown[]) => rejectInvitationMock(...args),
  openInvitationsSocket: (...args: unknown[]) => openInvitationsSocketMock(...args),
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => ({
    user: {
      id: "user-1",
      username: "dieg0",
      email: "dieg0@example.com",
      is_active: true,
      is_admin: false,
      created_at: "2026-02-22T00:00:00Z",
      updated_at: "2026-02-22T00:00:00Z",
    },
    loading: false,
    isAuthenticated: true,
    accessToken: "token-123",
    register: vi.fn(),
    login: vi.fn(),
    logout: vi.fn(),
    refreshUser: vi.fn(),
  }),
}));

describe("ProfilePage", () => {
  beforeEach(() => {
    fetchMyGamesMock.mockReset();
    deleteMyGameMock.mockReset();
    fetchActiveSeasonMock.mockReset();
    fetchUserRatingMock.mockReset();
    fetchRatingEventsMock.mockReset();
    fetchIncomingInvitationsMock.mockReset();
    acceptInvitationMock.mockReset();
    rejectInvitationMock.mockReset();
    openInvitationsSocketMock.mockReset();

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
      rating: 1242.4,
      games_played: 10,
      wins: 6,
      losses: 3,
      draws: 1,
      updated_at: "2026-02-22T00:00:00Z",
      league: "Protocol",
      division: "II",
      lp: 42,
      next_major_promo: null,
    });
    fetchRatingEventsMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 6,
      offset: 0,
      has_more: false,
    });
    fetchIncomingInvitationsMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 12,
      offset: 0,
      has_more: false,
    });
    acceptInvitationMock.mockResolvedValue({});
    rejectInvitationMock.mockResolvedValue({});
    openInvitationsSocketMock.mockReturnValue({
      close: vi.fn(),
      onerror: null,
      onmessage: null,
      onclose: null,
    });
  });

  it("loads and renders profile history list", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [
        {
          id: "game-12345678",
          season_id: null,
          queue_type: "ranked",
          status: "finished",
          rated: true,
          player1_id: "user-1",
          player2_id: "user-2",
          player1_agent: "human",
          player2_agent: "model",
          model_version_id: null,
          winner_side: "player1",
          winner_user_id: "user-1",
          termination_reason: "normal_end",
          source: "web",
          quality_score: null,
          is_training_eligible: true,
        },
      ],
      total: 1,
      limit: 8,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(fetchMyGamesMock).toHaveBeenCalledWith("token-123", 8, 0, ["finished"]);
    });

    expect(screen.getByText("Mi perfil")).toBeInTheDocument();
    expect(screen.getByText("Estado competitivo")).toBeInTheDocument();
    expect(screen.getByText("Actividad de ladder")).toBeInTheDocument();
    expect(screen.getByText("dieg0")).toBeInTheDocument();
    expect(screen.getByText("Historial de partidas")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByLabelText("Resultado: Victoria")).toBeInTheDocument();
      expect(screen.getByLabelText("Estado: Finalizada")).toBeInTheDocument();
    });
  });

  it("deletes a replay from profile history", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [
        {
          id: "game-12345678",
          season_id: null,
          queue_type: "casual",
          status: "in_progress",
          rated: false,
          player1_id: "user-1",
          player2_id: null,
          player1_agent: "human",
          player2_agent: "model",
          model_version_id: null,
          winner_side: null,
          winner_user_id: null,
          termination_reason: null,
          source: "human",
          quality_score: null,
          is_training_eligible: false,
        },
      ],
      total: 1,
      limit: 8,
      offset: 0,
      has_more: false,
    });
    deleteMyGameMock.mockResolvedValue(undefined);

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByText("game-123")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Eliminar game-123" }));
    await waitFor(() => {
      expect(screen.getByText("Confirmar eliminacion")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: "Si, eliminar" }));

    await waitFor(() => {
      expect(deleteMyGameMock).toHaveBeenCalledWith("token-123", "game-12345678");
    });
  });

  it("closes delete modal on Escape without deleting", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [
        {
          id: "game-99999999",
          season_id: null,
          queue_type: "casual",
          status: "in_progress",
          rated: false,
          player1_id: "user-1",
          player2_id: null,
          player1_agent: "human",
          player2_agent: "model",
          model_version_id: null,
          winner_side: null,
          winner_user_id: null,
          termination_reason: null,
          source: "human",
          quality_score: null,
          is_training_eligible: false,
        },
      ],
      total: 1,
      limit: 8,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByText("game-999")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Eliminar game-999" }));
    await waitFor(() => {
      expect(screen.getByText("Confirmar eliminacion")).toBeInTheDocument();
    });

    fireEvent.keyDown(window, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByText("Confirmar eliminacion")).not.toBeInTheDocument();
    });
    expect(deleteMyGameMock).not.toHaveBeenCalled();
  });

  it("shows spectator outcome for IA vs IA games not played by the user", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [
        {
          id: "game-ai-view1",
          season_id: null,
          queue_type: "casual",
          status: "finished",
          rated: false,
          player1_id: "bot-1",
          player2_id: "bot-2",
          player1_agent: "heuristic",
          player2_agent: "heuristic",
          model_version_id: null,
          winner_side: "p2",
          winner_user_id: "bot-2",
          termination_reason: "normal",
          source: "human",
          quality_score: null,
          is_training_eligible: false,
        },
      ],
      total: 1,
      limit: 8,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByLabelText("Resultado: Resultado IA")).toBeInTheDocument();
      expect(screen.getByText("Casual - espectador")).toBeInTheDocument();
    });
  });

  it("does not show stable label when rating transition is neutral", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 8,
      offset: 0,
      has_more: false,
    });
    fetchRatingEventsMock.mockResolvedValue({
      items: [
        {
          id: "event-1",
          user_id: "user-1",
          season_id: "season-1",
          game_id: "game-1",
          delta: 5,
          reason: "ranked_match",
          before_rating: 1200,
          after_rating: 1205,
          before_league: "Protocol",
          before_division: "III",
          before_lp: 0,
          after_league: "Protocol",
          after_division: "III",
          after_lp: 5,
          transition_type: null,
          major_promo_name: null,
          created_at: "2026-02-22T00:00:00Z",
        },
      ],
      total: 1,
      limit: 6,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByText("LP de liga")).toBeInTheDocument();
      expect(screen.getByText("+5 LP")).toBeInTheDocument();
      expect(screen.getByText("MMR paralelo +5.0")).toBeInTheDocument();
    });
    expect(screen.queryByText("Estable")).not.toBeInTheDocument();
  });

  it("shows MMR label when LP values are not available", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 8,
      offset: 0,
      has_more: false,
    });
    fetchRatingEventsMock.mockResolvedValue({
      items: [
        {
          id: "event-2",
          user_id: "user-1",
          season_id: "season-1",
          game_id: "game-2",
          delta: -12.3,
          reason: "ranked_match",
          before_rating: 1210,
          after_rating: 1197.7,
          before_league: "Protocol",
          before_division: "III",
          before_lp: null,
          after_league: "Protocol",
          after_division: "III",
          after_lp: null,
          transition_type: null,
          major_promo_name: null,
          created_at: "2026-02-22T00:00:00Z",
        },
      ],
      total: 1,
      limit: 6,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByText("Delta MMR")).toBeInTheDocument();
      expect(screen.getByText("-12.3 MMR")).toBeInTheDocument();
    });
    expect(screen.queryByText(/LP de liga/i)).not.toBeInTheDocument();
  });

  it("renders pending invitations card", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 8,
      offset: 0,
      has_more: false,
    });
    fetchIncomingInvitationsMock.mockResolvedValue({
      items: [
        {
          id: "invite-abc12345",
          queue_type: "custom",
          status: "pending",
          player1_id: "user-2",
          player2_id: "user-1",
          created_by_user_id: "user-2",
          player1_agent: "human",
          player2_agent: "human",
          created_at: "2026-02-25T00:00:00Z",
        },
      ],
      total: 1,
      limit: 12,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByText("Invitaciones 1v1")).toBeInTheDocument();
      expect(screen.getByText("invite-a")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Aceptar" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Rechazar" })).toBeInTheDocument();
    });
  });

  it("rejects a pending invitation from profile", async () => {
    fetchMyGamesMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 8,
      offset: 0,
      has_more: false,
    });
    fetchIncomingInvitationsMock.mockResolvedValue({
      items: [
        {
          id: "invite-reject-1",
          queue_type: "custom",
          status: "pending",
          player1_id: "user-2",
          player2_id: "user-1",
          created_by_user_id: "user-2",
          player1_agent: "human",
          player2_agent: "human",
          created_at: "2026-02-25T00:00:00Z",
        },
      ],
      total: 1,
      limit: 12,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Rechazar" })).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: "Rechazar" }));

    await waitFor(() => {
      expect(rejectInvitationMock).toHaveBeenCalledWith("token-123", "invite-reject-1");
    });
  });

  it("updates invitations from websocket events", async () => {
    let wsHandler:
      | ((event: { type: string; payload?: { items: Array<{ id: string; status: string }> } }) => void)
      | null = null;
    openInvitationsSocketMock.mockImplementation((_token: string, onEvent: typeof wsHandler) => {
      wsHandler = onEvent;
      return {
        close: vi.fn(),
        onerror: null,
        onmessage: null,
        onclose: null,
      };
    });
    fetchMyGamesMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 8,
      offset: 0,
      has_more: false,
    });
    fetchIncomingInvitationsMock.mockResolvedValue({
      items: [],
      total: 0,
      limit: 12,
      offset: 0,
      has_more: false,
    });

    renderWithProviders(<ProfilePage />, { route: "/profile" });
    await act(async () => {
      wsHandler?.({
        type: "invitations.status",
        payload: { items: [{ id: "invite-live-01", status: "pending" }] },
      });
    });

    await waitFor(() => {
      expect(screen.getByText("invite-l")).toBeInTheDocument();
    });
  });
});
