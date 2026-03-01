import { act, fireEvent, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/widgets/layout/AppShell";
import { renderWithProviders } from "@/test/render";

const logoutMock = vi.fn();
const openInvitationsSocketMock = vi.fn();
const acceptInvitationMock = vi.fn();
const rejectInvitationMock = vi.fn();
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

vi.mock("@/features/matches/api", () => ({
  openInvitationsSocket: (...args: unknown[]) => openInvitationsSocketMock(...args),
  acceptInvitation: (...args: unknown[]) => acceptInvitationMock(...args),
  rejectInvitation: (...args: unknown[]) => rejectInvitationMock(...args),
}));

describe("AppShell", () => {
  beforeEach(() => {
    logoutMock.mockReset();
    openInvitationsSocketMock.mockReset();
    acceptInvitationMock.mockReset();
    rejectInvitationMock.mockReset();
    authState = {
      isAuthenticated: true,
      user: { username: "test" },
    };
    openInvitationsSocketMock.mockReturnValue({
      close: vi.fn(),
      onerror: null,
      onmessage: null,
      onclose: null,
    });
    acceptInvitationMock.mockResolvedValue({});
    rejectInvitationMock.mockResolvedValue({});
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

});
