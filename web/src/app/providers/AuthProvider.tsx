import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCurrentUser,
  loginUser,
  logoutUser,
  refreshTokens,
  registerUser,
  type AuthUser,
  type AuthTokens,
} from "@/features/auth/api";
import { AuthContext, type AuthContextValue } from "@/app/providers/auth-context";

const AUTH_STORAGE_KEY = "ataxx.auth.tokens";

type StoredTokens = {
  accessToken: string;
  refreshToken: string;
};

function loadTokens(): StoredTokens | null {
  const raw = localStorage.getItem(AUTH_STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as StoredTokens;
    if (!parsed.accessToken || !parsed.refreshToken) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function saveTokens(tokens: StoredTokens | null): void {
  if (tokens === null) {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(tokens));
}

function toStored(tokens: AuthTokens): StoredTokens {
  return {
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
  };
}

export function AuthProvider({ children }: { children: React.ReactNode }): JSX.Element {
  const [tokens, setTokens] = useState<StoredTokens | null>(() => loadTokens());
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const clearSession = useCallback(() => {
    setTokens(null);
    setUser(null);
    saveTokens(null);
  }, []);

  const refreshUser = useCallback(async () => {
    if (tokens === null) {
      setUser(null);
      return;
    }
    try {
      const me = await getCurrentUser(tokens.accessToken);
      setUser(me);
      return;
    } catch {
      // Fallback a refresh si access token expiró.
    }
    try {
      const next = await refreshTokens(tokens.refreshToken);
      const stored = toStored(next);
      setTokens(stored);
      saveTokens(stored);
      const me = await getCurrentUser(stored.accessToken);
      setUser(me);
    } catch {
      clearSession();
    }
  }, [clearSession, tokens]);

  useEffect(() => {
    void (async () => {
      try {
        await refreshUser();
      } finally {
        setLoading(false);
      }
    })();
  }, [refreshUser]);

  const login = useCallback(async (payload: { username_or_email: string; password: string }) => {
    const result = await loginUser(payload);
    const stored = toStored(result);
    setTokens(stored);
    saveTokens(stored);
    const me = await getCurrentUser(stored.accessToken);
    setUser(me);
  }, []);

  const register = useCallback(async (payload: { username: string; email?: string; password: string }) => {
    await registerUser(payload);
    await login({ username_or_email: payload.email ?? payload.username, password: payload.password });
  }, [login]);

  const logout = useCallback(async () => {
    if (tokens !== null) {
      try {
        await logoutUser(tokens.refreshToken);
      } catch {
        // No bloquear logout local si la API falla.
      }
    }
    clearSession();
  }, [clearSession, tokens]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      isAuthenticated: user !== null,
      accessToken: tokens?.accessToken ?? null,
      register,
      login,
      logout,
      refreshUser,
    }),
    [loading, login, logout, refreshUser, register, tokens?.accessToken, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
