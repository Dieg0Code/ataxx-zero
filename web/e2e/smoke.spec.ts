import { expect, test } from "@playwright/test";

test.describe("Ataxx web smoke", () => {
  test("landing renders main CTA", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Arena Ataxx")).toBeVisible();
    await expect(page.getByRole("button", { name: "Buscar partida" })).toBeVisible();
  });

  test("login page is reachable and renders auth fields", async ({ page }) => {
    await page.goto("/auth/login");
    await expect(page.getByText("Iniciar sesion")).toBeVisible();
    await expect(page.getByPlaceholder("usuario o correo")).toBeVisible();
    await expect(page.getByPlaceholder("password")).toBeVisible();
    await expect(page.locator("form").getByRole("button", { name: "Entrar" })).toBeVisible();
  });
});
