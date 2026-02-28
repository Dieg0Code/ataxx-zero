import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { apiGet } from "@/shared/api/client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api client auth refresh retry", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("retries once when backend returns 422 invalid/expired token", async () => {
    localStorage.setItem(
      "ataxx.auth.tokens",
      JSON.stringify({ accessToken: "old-access", refreshToken: "refresh-1" }),
    );

    const dispatchSpy = vi.spyOn(window, "dispatchEvent");
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ detail: "Invalid or expired token." }, 422))
      .mockResolvedValueOnce(
        jsonResponse({
          access_token: "new-access",
          refresh_token: "new-refresh",
          token_type: "bearer",
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));

    const result = await apiGet<{ ok: boolean }>("/api/v1/gameplay/games?limit=8&offset=0", {
      headers: { Authorization: "Bearer old-access" },
    });

    expect(result).toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/v1/auth/refresh",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/api/v1/gameplay/games?limit=8&offset=0",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer new-access" }),
      }),
    );
    expect(JSON.parse(localStorage.getItem("ataxx.auth.tokens") ?? "{}")).toEqual({
      accessToken: "new-access",
      refreshToken: "new-refresh",
    });
    expect(dispatchSpy).toHaveBeenCalled();
  });

  it("does not refresh for non-auth 422", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ detail: "validation failed" }, 422));

    await expect(
      apiGet<{ ok: boolean }>("/api/v1/gameplay/games?limit=8&offset=0", {
        headers: { Authorization: "Bearer any" },
      }),
    ).rejects.toThrow("validation failed");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("deduplicates concurrent refresh requests and retries both calls", async () => {
    localStorage.setItem(
      "ataxx.auth.tokens",
      JSON.stringify({ accessToken: "old-access", refreshToken: "refresh-1" }),
    );

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ detail: "Invalid or expired token." }, 422))
      .mockResolvedValueOnce(jsonResponse({ detail: "Invalid or expired token." }, 422))
      .mockResolvedValueOnce(
        jsonResponse({
          access_token: "new-access",
          refresh_token: "new-refresh",
          token_type: "bearer",
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }))
      .mockResolvedValueOnce(jsonResponse({ ok: true }));

    const [r1, r2] = await Promise.all([
      apiGet<{ ok: boolean }>("/api/v1/gameplay/games?limit=8&offset=0", {
        headers: { Authorization: "Bearer old-access" },
      }),
      apiGet<{ ok: boolean }>("/api/v1/gameplay/games?limit=8&offset=0", {
        headers: { Authorization: "Bearer old-access" },
      }),
    ]);

    expect(r1).toEqual({ ok: true });
    expect(r2).toEqual({ ok: true });

    const refreshCalls = fetchMock.mock.calls.filter((call) => call[0] === "/api/v1/auth/refresh");
    expect(refreshCalls).toHaveLength(1);
    expect(JSON.parse(localStorage.getItem("ataxx.auth.tokens") ?? "{}")).toEqual({
      accessToken: "new-access",
      refreshToken: "new-refresh",
    });
  });
});
