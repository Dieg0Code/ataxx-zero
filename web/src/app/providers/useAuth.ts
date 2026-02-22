import { useContext } from "react";
import { AuthContext, type AuthContextValue } from "@/app/providers/auth-context";

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used inside AuthProvider.");
  }
  return context;
}

