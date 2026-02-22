import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/app/providers/useAuth";

export function RequireAuth({ children }: { children: JSX.Element }): JSX.Element {
  const { loading, isAuthenticated } = useAuth();
  const location = useLocation();

  if (loading) {
    return <p className="px-4 py-8 text-sm text-textDim">Validando sesion...</p>;
  }
  if (!isAuthenticated) {
    return <Navigate to="/auth/login" replace state={{ from: location.pathname }} />;
  }
  return children;
}

export function RequireGuest({ children }: { children: JSX.Element }): JSX.Element {
  const { loading, isAuthenticated } = useAuth();
  if (loading) {
    return <p className="px-4 py-8 text-sm text-textDim">Validando sesion...</p>;
  }
  if (isAuthenticated) {
    return <Navigate to="/profile" replace />;
  }
  return children;
}
