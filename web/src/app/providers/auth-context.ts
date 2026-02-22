import { createContext } from "react";
import type { AuthUser } from "@/features/auth/api";

export type AuthContextValue = {
  user: AuthUser | null;
  loading: boolean;
  isAuthenticated: boolean;
  accessToken: string | null;
  register: (payload: { username: string; email?: string; password: string }) => Promise<void>;
  login: (payload: { username_or_email: string; password: string }) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
};

export const AuthContext = createContext<AuthContextValue | null>(null);

