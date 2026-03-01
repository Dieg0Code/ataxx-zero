import { expect, test, type Page, type Route } from "@playwright/test";

const ACCESS_TOKEN = "access-e2e-token";
const REFRESH_TOKEN = "refresh-e2e-token";

async function fulfillJson(route: Route, payload: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function mockApi(page: Page): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (path === "/api/v1/auth/login" && method === "POST") {
      await fulfillJson(route, {
        access_token: ACCESS_TOKEN,
        refresh_token: REFRESH_TOKEN,
        token_type: "bearer",
      });
      return;
    }

    if (path === "/api/v1/auth/me" && method === "GET") {
      await fulfillJson(route, {
        id: "user-e2e-1",
        username: "diego-e2e",
        email: "diego@example.com",
        is_active: true,
        is_admin: false,
        created_at: "2026-03-01T00:00:00Z",
        updated_at: "2026-03-01T00:00:00Z",
      });
      return;
    }

    if (path.startsWith("/api/v1/ranking/seasons/active") && method === "GET") {
      await fulfillJson(route, {
        id: "season-e2e-1",
        name: "S1",
        starts_at: "2026-01-01T00:00:00Z",
        ends_at: null,
        is_active: true,
      });
      return;
    }

    if (
      path.startsWith("/api/v1/ranking/ratings/user-e2e-1/season-e2e-1") &&
      !path.includes("/events") &&
      method === "GET"
    ) {
      await fulfillJson(route, {
        id: "rating-e2e-1",
        user_id: "user-e2e-1",
        season_id: "season-e2e-1",
        rating: 1500.0,
        games_played: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        updated_at: "2026-03-01T00:00:00Z",
        league: "Bronce",
        division: "IV",
        lp: 50,
        next_major_promo: null,
      });
      return;
    }

    if (path.startsWith("/api/v1/ranking/ratings/user-e2e-1/season-e2e-1/events") && method === "GET") {
      await fulfillJson(route, {
        items: [],
        total: 0,
        limit: 6,
        offset: 0,
        has_more: false,
      });
      return;
    }

    if (path.startsWith("/api/v1/gameplay/games") && method === "GET") {
      await fulfillJson(route, {
        items: [],
        total: 0,
        limit: 8,
        offset: 0,
        has_more: false,
      });
      return;
    }

    if (path.startsWith("/api/v1/matches/invitations/incoming") && method === "GET") {
      await fulfillJson(route, {
        items: [],
        total: 0,
        limit: 12,
        offset: 0,
        has_more: false,
      });
      return;
    }

    if (path === "/api/v1/auth/refresh" && method === "POST") {
      await fulfillJson(route, {
        access_token: ACCESS_TOKEN,
        refresh_token: REFRESH_TOKEN,
        token_type: "bearer",
      });
      return;
    }

    await fulfillJson(route, {});
  });
}

test.describe("Auth flow", () => {
  test("redirects guests from profile to login", async ({ page }) => {
    await page.goto("/profile");
    await expect(page).toHaveURL(/\/auth\/login$/);
    await expect(page.getByText("Iniciar sesion")).toBeVisible();
  });

  test("logs in and lands on profile", async ({ page }) => {
    await mockApi(page);
    await page.goto("/auth/login");
    await page.getByPlaceholder("usuario o correo").fill("diego-e2e");
    await page.getByPlaceholder("password").fill("Password123!");
    await page.locator("form").getByRole("button", { name: "Entrar" }).click();

    await expect(page).toHaveURL(/\/profile$/);
    await expect(page.getByText("Mi perfil")).toBeVisible();
    await expect(page.getByRole("button", { name: "Salir (diego-e2e)" })).toBeVisible();

    const tokens = await page.evaluate(() => localStorage.getItem("ataxx.auth.tokens"));
    expect(tokens).toContain("access-e2e-token");
    expect(tokens).toContain("refresh-e2e-token");
  });
});
