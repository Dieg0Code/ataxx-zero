import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BoardState } from "@/features/match/types";
import { MatchPage } from "@/pages/match/MatchPage";

const predictAIMoveMock = vi.fn();
const createPersistedGameMock = vi.fn();
const deletePersistedGameMock = vi.fn();
const fetchPersistedGameSummaryMock = vi.fn();
const fetchPersistedReplayMock = vi.fn();
const openPersistedGameSocketMock = vi.fn();
const fetchPublicPlayersMock = vi.fn();
const storeInferredMoveMock = vi.fn();
const storeManualMoveMock = vi.fn();
const useAuthMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/features/match/api", () => ({
  predictAIMove: (...args: unknown[]) => predictAIMoveMock(...args),
}));

vi.mock("@/features/match/persistence", () => ({
  createPersistedGame: (...args: unknown[]) => createPersistedGameMock(...args),
  deletePersistedGame: (...args: unknown[]) => deletePersistedGameMock(...args),
  fetchPersistedGameSummary: (...args: unknown[]) => fetchPersistedGameSummaryMock(...args),
  fetchPersistedReplay: (...args: unknown[]) => fetchPersistedReplayMock(...args),
  openPersistedGameSocket: (...args: unknown[]) => openPersistedGameSocketMock(...args),
  storeInferredMove: (...args: unknown[]) => storeInferredMoveMock(...args),
  storeManualMove: (...args: unknown[]) => storeManualMoveMock(...args),
}));

vi.mock("@/features/identity/api", () => ({
  fetchPublicPlayers: (...args: unknown[]) => fetchPublicPlayersMock(...args),
}));

vi.mock("react-router-dom", () => ({
  Link: ({ to, children, ...props }: { to: string; children: ReactNode }) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
  useNavigate: () => vi.fn(),
  useLocation: () => ({ pathname: "/match", search: "", hash: "", state: null }),
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => useAuthMock(),
}));

function buildPrediction(board: BoardState): { move: null; value: number; board_after: BoardState } {
  return {
    move: null,
    value: 0,
    board_after: board,
  };
}

describe("MatchPage spectator mode", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    predictAIMoveMock.mockReset();
    createPersistedGameMock.mockReset();
    deletePersistedGameMock.mockReset();
    fetchPersistedGameSummaryMock.mockReset();
    fetchPersistedReplayMock.mockReset();
    openPersistedGameSocketMock.mockReset();
    fetchPublicPlayersMock.mockReset();
    storeInferredMoveMock.mockReset();
    storeManualMoveMock.mockReset();
    openPersistedGameSocketMock.mockReturnValue({
      close: vi.fn(),
      onclose: null,
      onmessage: null,
    });
    fetchPersistedGameSummaryMock.mockResolvedValue({
      id: "game-keep",
      queue_type: "ranked",
      status: "in_progress",
      rated: true,
      player1_id: "u1",
      player2_id: "bot-p1",
      player1_agent: "human",
      player2_agent: "heuristic",
    });
    fetchPublicPlayersMock.mockResolvedValue([
      {
        user_id: "bot-p1",
        is_bot: true,
        username: "CipherNovice",
        bot_kind: "heuristic",
        agent_type: "heuristic",
        heuristic_level: "easy",
        model_mode: null,
        enabled: true,
      },
      {
        user_id: "bot-p2",
        is_bot: true,
        username: "KernelWarden",
        bot_kind: "heuristic",
        agent_type: "heuristic",
        heuristic_level: "hard",
        model_mode: null,
        enabled: true,
      },
    ]);
    useAuthMock.mockReturnValue({
      user: { id: "spectator", username: "spec" },
      loading: false,
      isAuthenticated: true,
      accessToken: "token-spec",
      register: vi.fn(),
      login: vi.fn(),
      logout: vi.fn(),
      refreshUser: vi.fn(),
    });
  });

  it("does not auto-start in spectate mode and allows pre-start configuration", async () => {
    render(<MatchPage />);

    fireEvent.change(screen.getByLabelText("Modo de partida"), {
      target: { value: "spectate" },
    });

    expect(screen.getByRole("button", { name: "Iniciar partida" })).toBeInTheDocument();
    expect(screen.getByLabelText("Jugador P1")).toBeEnabled();
    expect(screen.getByLabelText("Jugador P2")).toBeEnabled();
    expect(screen.getByRole("button", { name: "Swap P1/P2" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Aleatorio" })).toBeInTheDocument();

    await waitFor(
      () => {
        expect(predictAIMoveMock).not.toHaveBeenCalled();
      },
      { timeout: 400 },
    );
  }, 15000);

  it("locks AI selectors once spectate match starts", async () => {
    predictAIMoveMock.mockImplementation(async (board: BoardState) => buildPrediction(board));
    render(<MatchPage />);

    fireEvent.change(screen.getByLabelText("Modo de partida"), {
      target: { value: "spectate" },
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Jugador P2")).toHaveValue("bot-p1");
    });

    fireEvent.click(screen.getByRole("button", { name: "Iniciar partida" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Partida en curso" })).toBeInTheDocument();
    });

    expect(screen.getByLabelText("Modo de partida")).toBeDisabled();
    expect(screen.getByLabelText("Jugador P1")).toBeDisabled();
    expect(screen.getByLabelText("Jugador P2")).toBeDisabled();
  }, 15000);
});

describe("MatchPage automatic persistence", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    predictAIMoveMock.mockReset();
    createPersistedGameMock.mockReset();
    deletePersistedGameMock.mockReset();
    fetchPersistedGameSummaryMock.mockReset();
    fetchPersistedReplayMock.mockReset();
    openPersistedGameSocketMock.mockReset();
    fetchPublicPlayersMock.mockReset();
    storeInferredMoveMock.mockReset();
    storeManualMoveMock.mockReset();
    openPersistedGameSocketMock.mockReturnValue({
      close: vi.fn(),
      onclose: null,
      onmessage: null,
    });
    fetchPersistedGameSummaryMock.mockResolvedValue({
      id: "game-keep",
      queue_type: "ranked",
      status: "in_progress",
      rated: true,
      player1_id: "u1",
      player2_id: "bot-p1",
      player1_agent: "human",
      player2_agent: "heuristic",
    });
    fetchPublicPlayersMock.mockResolvedValue([
      {
        user_id: "bot-p1",
        is_bot: true,
        username: "CipherNovice",
        bot_kind: "heuristic",
        agent_type: "heuristic",
        heuristic_level: "easy",
        model_mode: null,
        enabled: true,
      },
      {
        user_id: "bot-p2",
        is_bot: true,
        username: "KernelWarden",
        bot_kind: "heuristic",
        agent_type: "heuristic",
        heuristic_level: "hard",
        model_mode: null,
        enabled: true,
      },
    ]);
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
  });

  it("activates remote persistence automatically on match start", async () => {
    createPersistedGameMock.mockResolvedValue("game-123");
    render(<MatchPage />);

    await waitFor(() => {
      expect(screen.getByLabelText("Rival (jugador)")).toHaveValue("bot-p1");
    });

    fireEvent.click(screen.getByRole("button", { name: "Iniciar partida" }));

    await waitFor(() => {
      expect(createPersistedGameMock).toHaveBeenCalledWith(
        "token-123",
        "heuristic",
        expect.objectContaining({
          player1Agent: "human",
          player2Agent: "heuristic",
          ranked: false,
          preferredHeuristicLevel: "easy",
          preferredModelMode: "fast",
          selectedP2BotUserId: "bot-p1",
        }),
      );
    });

    expect(screen.getByText(/Guardado remoto activo/i)).toBeInTheDocument();
  }, 15000);

  it("forces casual persistence in spectate IA vs IA", async () => {
    createPersistedGameMock.mockResolvedValue("game-ranked");
    render(<MatchPage />);

    fireEvent.change(screen.getByLabelText("Modo de partida"), {
      target: { value: "spectate" },
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Jugador P1")).toHaveValue("bot-p1");
      expect(screen.getByLabelText("Jugador P2")).toHaveValue("bot-p1");
    });

    fireEvent.change(screen.getByLabelText("Jugador P1"), {
      target: { value: "bot-p1" },
    });
    fireEvent.change(screen.getByLabelText("Jugador P2"), {
      target: { value: "bot-p2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Iniciar partida" }));

    await waitFor(() => {
      expect(createPersistedGameMock).toHaveBeenCalledWith(
        "token-123",
        "heuristic",
        expect.objectContaining({
          ranked: false,
          selectedP1BotUserId: "bot-p1",
          selectedP2BotUserId: "bot-p2",
          player1Agent: "heuristic",
          player2Agent: "heuristic",
        }),
      );
    });
  }, 15000);

  it("does not delete active persisted game when auth token refresh triggers rerender", async () => {
    createPersistedGameMock.mockResolvedValue("game-keepalive");
    const logoutMock = vi.fn();
    const authState: {
      user: { id: string; username: string };
      loading: boolean;
      isAuthenticated: boolean;
      accessToken: string | null;
      register: ReturnType<typeof vi.fn>;
      login: ReturnType<typeof vi.fn>;
      logout: ReturnType<typeof vi.fn>;
      refreshUser: ReturnType<typeof vi.fn>;
    } = {
      user: { id: "u1", username: "demo" },
      loading: false,
      isAuthenticated: true,
      accessToken: "token-123",
      register: vi.fn(),
      login: vi.fn(),
      logout: logoutMock,
      refreshUser: vi.fn(),
    };
    useAuthMock.mockImplementation(() => authState);

    const view = render(<MatchPage />);
    await waitFor(() => {
      expect(screen.getByLabelText("Rival (jugador)")).toHaveValue("bot-p1");
    });
    fireEvent.click(screen.getByRole("button", { name: "Iniciar partida" }));

    await waitFor(() => {
      expect(createPersistedGameMock).toHaveBeenCalledWith(
        "token-123",
        "heuristic",
        expect.objectContaining({
          player1Agent: "human",
          player2Agent: "heuristic",
        }),
      );
    });

    authState.accessToken = "token-456";
    view.rerender(<MatchPage />);

    await waitFor(() => {
      expect(deletePersistedGameMock).not.toHaveBeenCalled();
    });
  }, 15000);
});
