import { apiGet, apiPost, apiPostNoContent } from "@/shared/api/client";

export type AuthUser = {
  id: string;
  username: string;
  email: string | null;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
  updated_at: string;
};

export type AuthTokens = {
  access_token: string;
  refresh_token: string;
  token_type: string;
};

export type RegisterPayload = {
  username: string;
  email?: string;
  password: string;
};

export type LoginPayload = {
  username_or_email: string;
  password: string;
};

export async function registerUser(payload: RegisterPayload): Promise<AuthUser> {
  return apiPost<AuthUser, RegisterPayload>("/api/v1/auth/register", payload);
}

export async function loginUser(payload: LoginPayload): Promise<AuthTokens> {
  return apiPost<AuthTokens, LoginPayload>("/api/v1/auth/login", payload);
}

export async function refreshTokens(refreshToken: string): Promise<AuthTokens> {
  return apiPost<AuthTokens, { refresh_token: string }>("/api/v1/auth/refresh", {
    refresh_token: refreshToken,
  });
}

export async function logoutUser(refreshToken: string): Promise<void> {
  return apiPostNoContent("/api/v1/auth/logout", { refresh_token: refreshToken });
}

export async function getCurrentUser(accessToken: string): Promise<AuthUser> {
  return apiGet<AuthUser>("/api/v1/auth/me", {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
}

