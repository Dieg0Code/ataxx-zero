import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RegisterPage } from "@/pages/auth/RegisterPage";
import { renderWithProviders } from "@/test/render";

const registerMock = vi.fn();
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
    register: (...args: unknown[]) => registerMock(...args),
    login: vi.fn(),
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

describe("RegisterPage", () => {
  beforeEach(() => {
    registerMock.mockReset();
    navigateMock.mockReset();
  });

  it("blocks submit when passwords do not match", async () => {
    renderWithProviders(<RegisterPage />, { route: "/auth/register" });

    fireEvent.change(screen.getByPlaceholderText("usuario"), {
      target: { value: "dieg0" },
    });
    fireEvent.change(screen.getByPlaceholderText("password"), {
      target: { value: "password123" },
    });
    fireEvent.change(screen.getByPlaceholderText("repite password"), {
      target: { value: "password321" },
    });
    fireEvent.click(screen.getAllByRole("button", { name: "Crear cuenta" }).at(-1)!);

    await waitFor(() => {
      expect(screen.getAllByText("Las passwords no coinciden.").length).toBeGreaterThan(0);
    });
    expect(registerMock).not.toHaveBeenCalled();
  });

  it("registers and redirects to profile when form is valid", async () => {
    registerMock.mockResolvedValue(undefined);
    renderWithProviders(<RegisterPage />, { route: "/auth/register" });

    fireEvent.change(screen.getByPlaceholderText("usuario"), {
      target: { value: "dieg0" },
    });
    fireEvent.change(screen.getByPlaceholderText("correo (opcional)"), {
      target: { value: "dieg0@example.com" },
    });
    fireEvent.change(screen.getByPlaceholderText("password"), {
      target: { value: "password123" },
    });
    fireEvent.change(screen.getByPlaceholderText("repite password"), {
      target: { value: "password123" },
    });
    fireEvent.click(screen.getAllByRole("button", { name: "Crear cuenta" }).at(-1)!);

    await waitFor(() => {
      expect(registerMock).toHaveBeenCalledWith({
        username: "dieg0",
        email: "dieg0@example.com",
        password: "password123",
      });
      expect(navigateMock).toHaveBeenCalledWith(
        "/profile",
        expect.objectContaining({
          replace: true,
          state: expect.objectContaining({
            flash: expect.objectContaining({
              message: expect.stringContaining("Cuenta creada"),
              tone: "success",
            }),
          }),
        }),
      );
    });
  });
});
