import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/widgets/layout/AppShell";
import { renderWithProviders } from "@/test/render";

const logoutMock = vi.fn();
let authState = {
  isAuthenticated: true,
  user: { username: "test" },
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

describe("AppShell", () => {
  beforeEach(() => {
    logoutMock.mockReset();
    authState = {
      isAuthenticated: true,
      user: { username: "test" },
    };
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

});
