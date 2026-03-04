import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LobbyPage } from "@/pages/lobby/LobbyPage";
import { renderWithProviders } from "@/test/render";

const prefetchPublicPlayersMock = vi.fn();
const useAuthMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/features/identity/api", () => ({
  prefetchPublicPlayers: (...args: unknown[]) => prefetchPublicPlayersMock(...args),
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => useAuthMock(),
}));

describe("LobbyPage", () => {
  beforeEach(() => {
    prefetchPublicPlayersMock.mockReset();
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

  it("shows direct links for invite, bot and spectate flows", () => {
    renderWithProviders(<LobbyPage />, { route: "/lobby" });

    expect(screen.getByRole("link", { name: /abrir sala 1v1/i })).toHaveAttribute("href", "/match?setup=invite");
    expect(screen.getByRole("link", { name: /crear duelo vs bot/i })).toHaveAttribute("href", "/match?setup=bot");
    expect(screen.getByRole("link", { name: /lanzar simulacion/i })).toHaveAttribute("href", "/match?setup=spectate");
  });

  it("prefetches player catalog when hovering setup links", async () => {
    renderWithProviders(<LobbyPage />, { route: "/lobby" });
    fireEvent.mouseEnter(screen.getByRole("link", { name: /abrir sala 1v1/i }));

    await waitFor(() => {
      expect(prefetchPublicPlayersMock).toHaveBeenCalledWith("token-123", { limit: 200 });
    });
  });
});
