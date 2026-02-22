import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AppShell } from "@/widgets/layout/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Button } from "@/shared/ui/button";
import { useAuth } from "@/app/providers/useAuth";

export function RegisterPage(): JSX.Element {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  async function onSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Las contraseñas no coinciden.");
      return;
    }
    setLoading(true);
    try {
      await register({ username, email: email.trim() ? email : undefined, password });
      navigate("/profile", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo registrar la cuenta.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-md">
        <Card>
          <CardHeader>
            <CardTitle>Crear cuenta</CardTitle>
            <CardDescription>Registra tu perfil para guardar partidas y competir en ranking.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-3" onSubmit={onSubmit}>
              <Input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="usuario" required />
              <Input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                type="email"
                placeholder="correo (opcional)"
              />
              <Input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type="password"
                placeholder="password"
                minLength={8}
                required
              />
              <Input
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                type="password"
                placeholder="repite password"
                minLength={8}
                required
              />
              {error !== null ? <p className="text-sm text-redGlow">{error}</p> : null}
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Creando..." : "Crear cuenta"}
              </Button>
            </form>
            <p className="mt-3 text-sm text-textDim">
              ¿Ya tienes cuenta?{" "}
              <Link className="text-primary hover:underline" to="/auth/login">
                Inicia sesion
              </Link>
            </p>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
