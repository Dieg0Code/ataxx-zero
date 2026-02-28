type RequestOptions = {
  headers?: Record<string, string>;
};

type ErrorPayload = {
  message?: string;
  detail?: string;
};

const AUTH_STORAGE_KEY = "ataxx.auth.tokens";
export const AUTH_TOKENS_UPDATED_EVENT = "ataxx.auth.tokens.updated";
let refreshInFlight: Promise<string | null> | null = null;

function isAuthEndpoint(path: string): boolean {
  return path.startsWith("/api/v1/auth/");
}

function isAuthFailure(response: Response, payload: ErrorPayload): boolean {
  const detail = (payload.detail ?? payload.message ?? "").toLowerCase();
  if (response.status === 401) {
    return true;
  }
  return response.status === 422 && detail.includes("invalid or expired token");
}

function hasBearerAuth(init: RequestInit): boolean {
  const headers = (init.headers ?? {}) as Record<string, string>;
  return typeof headers.Authorization === "string" && headers.Authorization.startsWith("Bearer ");
}

function readStoredRefreshToken(): string | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as { refreshToken?: string };
    return typeof parsed.refreshToken === "string" && parsed.refreshToken.length > 0
      ? parsed.refreshToken
      : null;
  } catch {
    return null;
  }
}

async function refreshAccessTokenInternal(): Promise<string | null> {
  const refreshToken = readStoredRefreshToken();
  if (refreshToken === null) {
    return null;
  }
  const response = await fetch("/api/v1/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!response.ok) {
    return null;
  }
  const payload = (await response.json().catch(() => null)) as
    | { access_token?: string; refresh_token?: string }
    | null;
  if (!payload?.access_token || !payload?.refresh_token) {
    return null;
  }
  try {
    const nextTokens = {
      accessToken: payload.access_token,
      refreshToken: payload.refresh_token,
    };
    localStorage.setItem(
      AUTH_STORAGE_KEY,
      JSON.stringify(nextTokens),
    );
    window.dispatchEvent(
      new CustomEvent(AUTH_TOKENS_UPDATED_EVENT, {
        detail: nextTokens,
      }),
    );
  } catch {
    return null;
  }
  return payload.access_token;
}

async function refreshAccessToken(): Promise<string | null> {
  if (refreshInFlight !== null) {
    return refreshInFlight;
  }
  refreshInFlight = refreshAccessTokenInternal().finally(() => {
    refreshInFlight = null;
  });
  return refreshInFlight;
}

function withRefreshedBearer(init: RequestInit, accessToken: string): RequestInit {
  const headers = { ...((init.headers ?? {}) as Record<string, string>) };
  headers.Authorization = `Bearer ${accessToken}`;
  return { ...init, headers };
}

async function doRequest(path: string, init: RequestInit, retried = false): Promise<Response> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as ErrorPayload;
    const canRetry =
      !retried && !isAuthEndpoint(path) && hasBearerAuth(init) && isAuthFailure(response, payload);
    if (canRetry) {
      const nextAccessToken = await refreshAccessToken();
      if (nextAccessToken !== null) {
        return doRequest(path, withRefreshedBearer(init, nextAccessToken), true);
      }
    }
    throw new Error(payload.message ?? payload.detail ?? `HTTP ${response.status}`);
  }
  return response;
}

export async function apiGet<T>(path: string, options?: RequestOptions): Promise<T> {
  const response = await doRequest(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  return (await response.json()) as T;
}

export async function apiPost<TResponse, TBody>(
  path: string,
  body: TBody,
  options?: RequestOptions,
): Promise<TResponse> {
  const response = await doRequest(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    body: JSON.stringify(body),
  });
  return (await response.json()) as TResponse;
}

export async function apiPostNoContent<TBody>(
  path: string,
  body: TBody,
  options?: RequestOptions,
): Promise<void> {
  await doRequest(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    body: JSON.stringify(body),
  });
}

export async function apiDeleteNoContent(path: string, options?: RequestOptions): Promise<void> {
  await doRequest(path, {
    method: "DELETE",
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
}
