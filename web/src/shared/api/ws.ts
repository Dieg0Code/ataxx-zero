function toWsOrigin(rawBase: string): string {
  const trimmed = rawBase.trim().replace(/\/+$/, "");
  if (trimmed.startsWith("ws://") || trimmed.startsWith("wss://")) {
    return trimmed;
  }
  if (trimmed.startsWith("http://")) {
    return `ws://${trimmed.slice("http://".length)}`;
  }
  if (trimmed.startsWith("https://")) {
    return `wss://${trimmed.slice("https://".length)}`;
  }
  return trimmed;
}

function resolveWsOrigin(): string {
  const fromEnv = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  if (typeof fromEnv === "string" && fromEnv.trim().length > 0) {
    return toWsOrigin(fromEnv);
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname.toLowerCase();
  const isLocalDev = host === "localhost" || host === "127.0.0.1";
  if (isLocalDev) {
    return `${protocol}://127.0.0.1:8000`;
  }
  return `${protocol}://${window.location.host}`;
}

export function buildWsUrl(path: string, params?: Record<string, string>): string {
  const base = resolveWsOrigin();
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (params === undefined) {
    return `${base}${normalizedPath}`;
  }
  const searchParams = new URLSearchParams(params);
  return `${base}${normalizedPath}?${searchParams.toString()}`;
}

