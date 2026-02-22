import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AppShell } from "@/widgets/layout/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Button } from "@/shared/ui/button";
import { useAuth } from "@/app/providers/useAuth";

type LocationState = {
  from?: string;
};

export function LoginPage(): JSX.Element {
  const [usernameOrEmail, setUsernameOrEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as LocationState | null)?.from ?? "/profile";

  async function onSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login({ username_or_email: usernameOrEmail, password });
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo iniciar sesion.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-md">
        <Card>
          <CardHeader>
            <CardTitle>Iniciar sesion</CardTitle>
            <CardDescription>Accede a tu cuenta para jugar, guardar progreso y subir en ranking.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-3" onSubmit={onSubmit}>
              <Input
                value={usernameOrEmail}
                onChange={(e) => setUsernameOrEmail(e.target.value)}
                placeholder="usuario o correo"
                required
              />
              <Input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type="password"
                placeholder="password"
                required
              />
              {error !== null ? <p className="text-sm text-redGlow">{error}</p> : null}
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Entrando..." : "Entrar"}
              </Button>
            </form>
            <p className="mt-3 text-sm text-textDim">
              ¿Aun no tienes cuenta?{" "}
              <Link className="text-primary hover:underline" to="/auth/register">
                Registrate
              </Link>
            </p>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
