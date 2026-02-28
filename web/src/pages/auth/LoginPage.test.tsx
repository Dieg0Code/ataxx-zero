import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LoginPage } from "@/pages/auth/LoginPage";
import { renderWithProviders } from "@/test/render";

const loginMock = vi.fn();
const navigateMock = vi.fn();

vi.mock("@/widgets/layout/AppShell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/app/providers/useAuth", () => ({
  useAuth: () => ({
    user: null,
    loading: false,
    isAuthenticated: false,
    accessToken: null,
    register: vi.fn(),
    login: (...args: unknown[]) => loginMock(...args),
    logout: vi.fn(),
    refreshUser: vi.fn(),
  }),
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

describe("LoginPage", () => {
  beforeEach(() => {
    loginMock.mockReset();
    navigateMock.mockReset();
  });

  it("submits credentials and redirects to profile on success", async () => {
    loginMock.mockResolvedValue(undefined);
    renderWithProviders(<LoginPage />, { route: "/auth/login" });

    fireEvent.change(screen.getByPlaceholderText("usuario o correo"), {
      target: { value: "dieg0" },
    });
    fireEvent.change(screen.getByPlaceholderText("password"), {
      target: { value: "supersecret" },
    });
    fireEvent.click(screen.getAllByRole("button", { name: "Entrar" }).at(-1)!);

    await waitFor(() => {
      expect(loginMock).toHaveBeenCalledWith({
        username_or_email: "dieg0",
        password: "supersecret",
      });
      expect(navigateMock).toHaveBeenCalledWith(
        "/profile",
        expect.objectContaining({
          replace: true,
          state: expect.objectContaining({
            flash: expect.objectContaining({
              message: expect.stringContaining("Bienvenido"),
              tone: "success",
            }),
          }),
        }),
      );
    });
  });

  it("shows backend error if login fails", async () => {
    loginMock.mockRejectedValue(new Error("Credenciales invalidas"));
    renderWithProviders(<LoginPage />, { route: "/auth/login" });

    fireEvent.change(screen.getByPlaceholderText("usuario o correo"), {
      target: { value: "dieg0" },
    });
    fireEvent.change(screen.getByPlaceholderText("password"), {
      target: { value: "bad" },
    });
    fireEvent.click(screen.getAllByRole("button", { name: "Entrar" }).at(-1)!);

    await waitFor(() => {
      expect(screen.getByText("Credenciales invalidas")).toBeInTheDocument();
      expect(navigateMock).not.toHaveBeenCalled();
    });
  });
});
