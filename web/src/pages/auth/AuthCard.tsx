import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Loader2, ShieldAlert } from "lucide-react";
import { useAuth } from "@/app/providers/useAuth";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";

type AuthTab = "login" | "register";

type AuthCardProps = {
  initialTab: AuthTab;
  from: string;
};

const RESERVED_USERNAMES = new Set([
  "admin",
  "root",
  "system",
  "support",
  "ataxx",
  "underbytelabs",
]);

function scorePassword(password: string): { score: number; label: string } {
  let score = 0;
  if (password.length >= 8) {
    score += 1;
  }
  if (/[A-Z]/.test(password)) {
    score += 1;
  }
  if (/[0-9]/.test(password)) {
    score += 1;
  }
  if (/[^A-Za-z0-9]/.test(password)) {
    score += 1;
  }
  if (score <= 1) {
    return { score, label: "Debil" };
  }
  if (score <= 3) {
    return { score, label: "Media" };
  }
  return { score, label: "Fuerte" };
}

export function AuthCard({ initialTab, from }: AuthCardProps): JSX.Element {
  const [tab, setTab] = useState<AuthTab>(initialTab);
  const [loginIdentifier, setLoginIdentifier] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [registerUsername, setRegisterUsername] = useState("");
  const [registerEmail, setRegisterEmail] = useState("");
  const [registerPassword, setRegisterPassword] = useState("");
  const [registerConfirmPassword, setRegisterConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const redirectTimerRef = useRef<number | null>(null);
  const { login, register } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    return () => {
      if (redirectTimerRef.current !== null) {
        window.clearTimeout(redirectTimerRef.current);
      }
    };
  }, []);

  const usernameState = useMemo(() => {
    const username = registerUsername.trim().toLowerCase();
    if (username.length === 0) {
      return { tone: "muted", text: "Elige un nombre de usuario." };
    }
    if (username.length < 3) {
      return { tone: "warn", text: "Minimo 3 caracteres." };
    }
    if (!/^[a-zA-Z0-9_]+$/.test(username)) {
      return { tone: "warn", text: "Usa letras, numeros o _." };
    }
    if (RESERVED_USERNAMES.has(username)) {
      return { tone: "error", text: "Ese nombre esta reservado." };
    }
    return { tone: "ok", text: "Disponible para crear cuenta." };
  }, [registerUsername]);

  const passwordStrength = useMemo(
    () => scorePassword(registerPassword),
    [registerPassword],
  );

  const confirmState = useMemo(() => {
    if (registerConfirmPassword.length === 0) {
      return { tone: "muted", text: "Repite la password." };
    }
    if (registerPassword !== registerConfirmPassword) {
      return { tone: "error", text: "Las passwords no coinciden." };
    }
    return { tone: "ok", text: "Passwords sincronizadas." };
  }, [registerConfirmPassword, registerPassword]);

  async function onLoginSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setSuccessMessage(null);
    setLoading(true);
    try {
      await login({ username_or_email: loginIdentifier, password: loginPassword });
      setSuccessMessage("Acceso concedido. Redirigiendo...");
      redirectTimerRef.current = window.setTimeout(() => {
        navigate(from, {
          replace: true,
          state: {
            flash: {
              message: `Bienvenido, ${loginIdentifier.trim() || "jugador"}.`,
              tone: "success",
            },
          },
        });
      }, 550);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo iniciar sesion.");
    } finally {
      setLoading(false);
    }
  }

  async function onRegisterSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (usernameState.tone !== "ok") {
      setError("Revisa el nombre de usuario.");
      return;
    }
    if (registerPassword !== registerConfirmPassword) {
      setError("Las passwords no coinciden.");
      return;
    }
    setLoading(true);
    try {
      await register({
        username: registerUsername,
        email: registerEmail.trim() ? registerEmail : undefined,
        password: registerPassword,
      });
      setSuccessMessage("Cuenta creada. Entrando a tu perfil...");
      redirectTimerRef.current = window.setTimeout(() => {
        navigate("/profile", {
          replace: true,
          state: {
            flash: {
              message: `Cuenta creada. Bienvenido, ${registerUsername.trim() || "jugador"}.`,
              tone: "success",
            },
          },
        });
      }, 550);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo crear la cuenta.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="border-zinc-800/90 bg-zinc-950/65 shadow-[0_0_26px_rgba(0,0,0,0.35)]">
      <CardHeader className="space-y-3">
        <div className="inline-flex rounded-md border border-zinc-700/80 bg-zinc-950/80 p-1">
          <button
            type="button"
            onClick={() => {
              setTab("login");
              setError(null);
              setSuccessMessage(null);
            }}
            disabled={loading}
            className={`relative rounded-sm px-3 py-1 text-xs font-medium transition ${
              tab === "login"
                ? "text-lime-200"
                : "text-zinc-400 hover:text-zinc-200"
            } ${loading ? "opacity-60" : ""}`}
          >
            {tab === "login" ? (
              <motion.span
                layoutId="auth-tab-indicator"
                className="absolute inset-0 rounded-sm border border-lime-300/40 bg-lime-300/10"
                transition={{ duration: 0.2, ease: "easeOut" }}
              />
            ) : null}
            <span className="relative">Entrar</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setTab("register");
              setError(null);
              setSuccessMessage(null);
            }}
            disabled={loading}
            className={`relative rounded-sm px-3 py-1 text-xs font-medium transition ${
              tab === "register"
                ? "text-lime-200"
                : "text-zinc-400 hover:text-zinc-200"
            } ${loading ? "opacity-60" : ""}`}
          >
            {tab === "register" ? (
              <motion.span
                layoutId="auth-tab-indicator"
                className="absolute inset-0 rounded-sm border border-lime-300/40 bg-lime-300/10"
                transition={{ duration: 0.2, ease: "easeOut" }}
              />
            ) : null}
            <span className="relative">Crear cuenta</span>
          </button>
        </div>
        <div>
          <CardTitle>{tab === "login" ? "Iniciar sesion" : "Registro rapido"}</CardTitle>
          <CardDescription>
            {tab === "login"
              ? "Entra para jugar, guardar progreso y competir."
              : "Crea tu cuenta y entra directo al ranking."}
          </CardDescription>
        </div>
      </CardHeader>

      <CardContent>
        <AnimatePresence mode="wait" initial={false}>
          {successMessage !== null ? (
            <motion.div
              key="auth-success"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              className="rounded-md border border-lime-300/35 bg-lime-300/10 px-3 py-2 text-sm text-lime-200"
            >
              <p className="inline-flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4" />
                {successMessage}
              </p>
            </motion.div>
          ) : tab === "login" ? (
            <motion.form
              key="auth-login"
              className="space-y-3"
              onSubmit={onLoginSubmit}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <Input
                value={loginIdentifier}
                onChange={(event) => setLoginIdentifier(event.target.value)}
                placeholder="usuario o correo"
                autoComplete="username"
                required
              />
              <Input
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
                type="password"
                placeholder="password"
                autoComplete="current-password"
                required
              />
              {error !== null ? <p className="text-sm text-redGlow">{error}</p> : null}
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Entrando...
                  </span>
                ) : (
                  "Entrar"
                )}
              </Button>
              <p className="text-xs text-textDim">
                No tienes cuenta?{" "}
                <Link className="text-primary hover:underline" to="/auth/register">
                  Registrate
                </Link>
              </p>
            </motion.form>
          ) : (
            <motion.form
              key="auth-register"
              className="space-y-3"
              onSubmit={onRegisterSubmit}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <Input
                value={registerUsername}
                onChange={(event) => setRegisterUsername(event.target.value)}
                placeholder="usuario"
                autoComplete="username"
                required
              />
              <p
                className={`inline-flex items-center gap-1 text-xs ${
                  usernameState.tone === "ok"
                    ? "text-lime-300"
                    : usernameState.tone === "error"
                      ? "text-redGlow"
                      : "text-zinc-400"
                }`}
              >
                {usernameState.tone === "ok" ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <ShieldAlert className="h-3.5 w-3.5" />
                )}
                {usernameState.text}
              </p>
              <Input
                value={registerEmail}
                onChange={(event) => setRegisterEmail(event.target.value)}
                type="email"
                placeholder="correo (opcional)"
                autoComplete="email"
              />
              <Input
                value={registerPassword}
                onChange={(event) => setRegisterPassword(event.target.value)}
                type="password"
                placeholder="password"
                minLength={8}
                autoComplete="new-password"
                required
              />
              <div className="space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                  <motion.div
                    className={`h-full ${
                      passwordStrength.score <= 1
                        ? "bg-red-400"
                        : passwordStrength.score <= 3
                          ? "bg-amber-300"
                          : "bg-lime-300"
                    }`}
                    initial={false}
                    animate={{ width: `${Math.max(8, (passwordStrength.score / 4) * 100)}%` }}
                    transition={{ duration: 0.25, ease: "easeOut" }}
                  />
                </div>
                <p className="text-xs text-zinc-400">Fuerza password: {passwordStrength.label}</p>
              </div>
              <Input
                value={registerConfirmPassword}
                onChange={(event) => setRegisterConfirmPassword(event.target.value)}
                type="password"
                placeholder="repite password"
                minLength={8}
                autoComplete="new-password"
                required
              />
              <p
                className={`inline-flex items-center gap-1 text-xs ${
                  confirmState.tone === "ok"
                    ? "text-lime-300"
                    : confirmState.tone === "error"
                      ? "text-redGlow"
                      : "text-zinc-400"
                }`}
              >
                {confirmState.tone === "ok" ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <ShieldAlert className="h-3.5 w-3.5" />
                )}
                {confirmState.text}
              </p>
              {error !== null ? <p className="text-sm text-redGlow">{error}</p> : null}
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Creando...
                  </span>
                ) : (
                  "Crear cuenta"
                )}
              </Button>
              <p className="text-xs text-textDim">
                Ya tienes cuenta?{" "}
                <Link className="text-primary hover:underline" to="/auth/login">
                  Inicia sesion
                </Link>
              </p>
            </motion.form>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
