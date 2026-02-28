import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PersistedReplay } from "@/features/match/persistence";
import { GameDetailPage } from "@/pages/profile/GameDetailPage";
import { buildBoardTimeline } from "@/pages/profile/replayTimeline";
import type { Cell } from "@/features/match/types";

const fetchPersistedReplayMock = vi.fn();
const useAuthMock = vi.fn();
const useParamsMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => useAuthMock(),
}));

vi.mock("react-router-dom", () => ({
  Link: ({ to, children, ...props }: { to: string; children: ReactNode }) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
  Navigate: ({ to }: { to: string }) => <div>navigate:{to}</div>,
  useParams: () => useParamsMock(),
}));

vi.mock("@/features/match/persistence", () => ({
  fetchPersistedReplay: (...args: unknown[]) => fetchPersistedReplayMock(...args),
}));

function renderPage(): void {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <GameDetailPage />
    </QueryClientProvider>,
  );
}

describe("GameDetailPage replay player", () => {
  beforeEach(() => {
    useAuthMock.mockReturnValue({
      accessToken: "token-123",
      loading: false,
      isAuthenticated: true,
    });
    useParamsMock.mockReturnValue({ gameId: "game-1" });
    fetchPersistedReplayMock.mockResolvedValue({
      game: {
        id: "game-1",
        status: "finished",
        winner_side: "p1",
        winner_user_id: "u1",
        termination_reason: "normal",
      },
      moves: [
        {
          ply: 0,
          player_side: "p1",
          r1: 0,
          c1: 0,
          r2: 1,
          c2: 1,
          board_before: { grid: Array.from({ length: 7 }, () => Array(7).fill(0)), current_player: 1, half_moves: 0 },
          board_after: { grid: Array.from({ length: 7 }, () => Array(7).fill(0)), current_player: -1, half_moves: 1 },
          mode: "manual",
        },
        {
          ply: 1,
          player_side: "p2",
          r1: 6,
          c1: 6,
          r2: 5,
          c2: 5,
          board_before: { grid: Array.from({ length: 7 }, () => Array(7).fill(0)), current_player: -1, half_moves: 1 },
          board_after: { grid: Array.from({ length: 7 }, () => Array(7).fill(0)), current_player: 1, half_moves: 2 },
          mode: "manual",
        },
      ],
    });
  });

  it("toggles replay playback controls", async () => {
    renderPage();

    expect(await screen.findByText("Paso 0/2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reproducir" }));
    expect(screen.getByRole("button", { name: "Pausar" })).toBeInTheDocument();
  });

  it("steps replay with arrow buttons", async () => {
    renderPage();

    expect(await screen.findByText("Paso 0/2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Paso siguiente" }));
    expect(screen.getByText("Paso 1/2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Paso anterior" }));
    expect(screen.getByText("Paso 0/2")).toBeInTheDocument();
  });
});

describe("buildBoardTimeline", () => {
  it("reconstructs board progression even if board_after snapshots are invalid", () => {
    const replay: PersistedReplay = {
      game: {
        id: "game-2",
        status: "finished",
        winner_side: "p1",
        winner_user_id: "u1",
        termination_reason: "normal",
      },
      moves: [
        {
          ply: 0,
          player_side: "p1",
          r1: 0,
          c1: 0,
          r2: 1,
          c2: 1,
          board_before: {
            grid: [
              [1, 0, 0, 0, 0, 0, -1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [-1, 0, 0, 0, 0, 0, 1],
            ],
            current_player: 1,
            half_moves: 0,
          },
          board_after: null,
          mode: "manual",
        },
        {
          ply: 1,
          player_side: "p2",
          r1: 0,
          c1: 6,
          r2: 1,
          c2: 5,
          board_before: null,
          board_after: null,
          mode: "manual",
        },
        {
          ply: 2,
          player_side: "p1",
          r1: 1,
          c1: 1,
          r2: 2,
          c2: 2,
          board_before: null,
          board_after: null,
          mode: "manual",
        },
        {
          ply: 3,
          player_side: "p2",
          r1: 1,
          c1: 5,
          r2: 2,
          c2: 4,
          board_before: null,
          board_after: null,
          mode: "manual",
        },
        {
          ply: 4,
          player_side: "p1",
          r1: 2,
          c1: 2,
          r2: 2,
          c2: 3,
          board_before: null,
          board_after: null,
          mode: "manual",
        },
      ],
    };

    const timeline = buildBoardTimeline(replay);

    expect(timeline).toHaveLength(6);
    expect(timeline[5].grid[2][4]).toBe(1);
  });

  it("handles omitted pass moves by aligning turn with player_side", () => {
    const fullRow: Cell[] = [1, 1, 1, 1, 1, 1, 1];
    const row = (...values: Cell[]): Cell[] => values;
    const replay: PersistedReplay = {
      game: {
        id: "game-pass",
        status: "finished",
        winner_side: "p1",
        winner_user_id: "u1",
        termination_reason: "normal",
      },
      moves: [
        {
          ply: 2,
          player_side: "p1",
          r1: 3,
          c1: 4,
          r2: 3,
          c2: 5,
          board_before: null,
          board_after: null,
          mode: "manual",
        },
        {
          ply: 1,
          player_side: "p1",
          r1: 3,
          c1: 3,
          r2: 3,
          c2: 4,
          board_before: {
            grid: [
              fullRow,
              fullRow,
              fullRow,
              row(1, 1, 1, 1, 0, 0, 1),
              fullRow,
              fullRow,
              fullRow,
            ],
            current_player: 1,
            half_moves: 0,
          },
          board_after: null,
          mode: "manual",
        },
      ],
    };

    const timeline = buildBoardTimeline(replay);

    expect(timeline).toHaveLength(3);
    expect(timeline[2].grid[3][5]).toBe(1);
  });
});
