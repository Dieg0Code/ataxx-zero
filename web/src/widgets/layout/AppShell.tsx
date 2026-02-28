import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Home,
  Info,
  LogOut,
  ShieldAlert,
  Trophy,
  User as UserIcon,
} from "lucide-react";
import { useAuth } from "@/app/providers/useAuth";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";
import { cn } from "@/shared/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Inicio", icon: Home },
  { to: "/ranking", label: "Ranking", icon: Trophy },
  { to: "/profile", label: "Perfil", icon: UserIcon },
];

type FlashTone = "success" | "warning" | "error" | "info";

type FlashMessage = {
  message: string;
  tone: FlashTone;
};

type FlashState = {
  flash?: string | { message: string; tone?: FlashTone };
};

function normalizeFlash(state: unknown): FlashMessage | null {
  if (typeof state !== "object" || state === null) {
    return null;
  }
  const maybeFlash = (state as FlashState).flash;
  if (typeof maybeFlash === "string" && maybeFlash.trim().length > 0) {
    return { message: maybeFlash, tone: "success" };
  }
  if (
    typeof maybeFlash === "object" &&
    maybeFlash !== null &&
    typeof maybeFlash.message === "string" &&
    maybeFlash.message.trim().length > 0
  ) {
    return {
      message: maybeFlash.message,
      tone: maybeFlash.tone ?? "info",
    };
  }
  return null;
}

function flashToneClass(tone: FlashTone): string {
  if (tone === "success") {
    return "border-lime-300/35 bg-zinc-950/92 text-lime-200";
  }
  if (tone === "warning") {
    return "border-lime-300/30 bg-zinc-950/92 text-lime-100";
  }
  if (tone === "error") {
    return "border-red-400/35 bg-zinc-950/92 text-red-200";
  }
  return "border-lime-300/30 bg-zinc-950/92 text-lime-100";
}

function FlashIcon({ tone }: { tone: FlashTone }): JSX.Element {
  if (tone === "success") {
    return <CheckCircle2 className="h-4 w-4" />;
  }
  if (tone === "warning") {
    return <AlertTriangle className="h-4 w-4" />;
  }
  if (tone === "error") {
    return <ShieldAlert className="h-4 w-4" />;
  }
  return <Info className="h-4 w-4" />;
}

type AppShellProps = {
  children: React.ReactNode;
  onNavigateAttempt?: (to: string) => boolean;
  onLogoutAttempt?: () => boolean;
};

export function AppShell({ children, onNavigateAttempt, onLogoutAttempt }: AppShellProps): JSX.Element {
  const { isAuthenticated, user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [flashMessage, setFlashMessage] = useState<FlashMessage | null>(null);

  useEffect(() => {
    const parsedFlash = normalizeFlash(location.state);
    if (parsedFlash === null) {
      return;
    }

    setFlashMessage(parsedFlash);
    navigate(`${location.pathname}${location.search}${location.hash}`, { replace: true, state: null });
  }, [location.hash, location.pathname, location.search, location.state, navigate]);

  useEffect(() => {
    if (flashMessage === null) {
      return;
    }
    const timer = window.setTimeout(() => {
      setFlashMessage(null);
    }, 3600);
    return () => {
      window.clearTimeout(timer);
    };
  }, [flashMessage]);

  return (
    <div className="mx-auto min-h-screen w-full max-w-[1320px] px-3 py-3 sm:px-5">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
        className="relative overflow-hidden rounded-2xl border border-zinc-800/90 bg-zinc-950/60 px-4 py-3 backdrop-blur-xl"
      >
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute -right-8 top-1 h-20 w-20 rounded-full bg-lime-300/10 blur-2xl"
          animate={{ opacity: [0.16, 0.32, 0.16], scale: [0.92, 1.06, 0.92] }}
          transition={{ duration: 4.2, repeat: Infinity, ease: "easeInOut" }}
        />

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <motion.div
              className="h-7 w-7 rounded-full bg-gradient-to-br from-lime-300 via-cyan-300 to-zinc-300 shadow-[0_0_18px_rgba(163,230,53,0.42)]"
              animate={{ scale: [1, 1.06, 1], opacity: [0.92, 1, 0.92] }}
              transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
            />
            <div>
              <p className="text-sm font-semibold text-textMain">underbyteLabs</p>
              <p className="text-xs text-textDim">ataxx-zero arena</p>
            </div>
            <Badge variant="success">beta</Badge>
          </div>

          <div className="flex items-center gap-2">
            {isAuthenticated ? (
              <Button
                size="sm"
                variant="ghost"
                className="border border-zinc-700/80 bg-zinc-950/65 text-zinc-200 hover:border-zinc-500/90 hover:bg-zinc-900/85 hover:text-white"
                onClick={() => {
                  if (onLogoutAttempt && !onLogoutAttempt()) {
                    return;
                  }
                  void logout();
                }}
              >
                <span className="inline-flex items-center gap-1.5">
                  <LogOut className="h-3.5 w-3.5 text-zinc-300" />
                  Salir ({user?.username ?? "user"})
                </span>
              </Button>
            ) : (
              <Button size="sm" asChild variant="secondary">
                <Link to="/auth/login">Entrar</Link>
              </Button>
            )}
          </div>
        </div>

        <nav className="mt-3 flex items-center gap-1 overflow-x-hidden overflow-y-hidden border-t border-zinc-800/80 pt-2">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.to;
            const Icon = item.icon;
            return (
              <Button
                key={item.to}
                asChild
                size="sm"
                variant="ghost"
                className={cn(
                  "relative rounded-md px-3 transition-all duration-200 hover:-translate-y-0.5 hover:text-lime-200",
                  active && "bg-lime-400/10 text-lime-200 shadow-[0_0_12px_rgba(163,230,53,0.12)]",
                  !active && "text-textDim",
                )}
              >
                <Link
                  to={item.to}
                  aria-current={active ? "page" : undefined}
                  className="inline-flex items-center gap-1.5"
                  onClick={(event) => {
                    if (!onNavigateAttempt) {
                      return;
                    }
                    const shouldProceed = onNavigateAttempt(item.to);
                    if (!shouldProceed) {
                      event.preventDefault();
                    }
                  }}
                >
                  <Icon className={cn("h-3.5 w-3.5 transition-colors", active ? "text-lime-300" : "text-textDim")} />
                  <span>{item.label}</span>
                  {active ? (
                    <motion.span
                      layoutId="nav-active-indicator"
                      className="absolute inset-x-2 bottom-0 h-px bg-gradient-to-r from-transparent via-lime-300 to-transparent"
                      transition={{ duration: 0.22, ease: "easeOut" }}
                    />
                  ) : null}
                </Link>
              </Button>
            );
          })}
        </nav>
      </motion.header>
      <AnimatePresence initial={false}>
        {flashMessage !== null ? (
          <motion.div
            key="appshell-flash"
            initial={{ opacity: 0, y: -8, height: 0, marginTop: 0, filter: "blur(2px)" }}
            animate={{
              opacity: 1,
              y: 0,
              height: "auto",
              marginTop: 12,
              filter: "blur(0px)",
              boxShadow: [
                "0 0 0px rgba(163,230,53,0.00)",
                "0 0 20px rgba(163,230,53,0.45)",
                "0 0 0px rgba(163,230,53,0.00)",
                "0 0 18px rgba(163,230,53,0.40)",
                "0 0 0px rgba(163,230,53,0.00)",
                "0 0 14px rgba(163,230,53,0.32)",
                "0 0 8px rgba(163,230,53,0.18)",
              ],
            }}
            exit={{ opacity: 0, y: -6, height: 0, marginTop: 0, filter: "blur(2px)" }}
            transition={{
              height: { duration: 0.34, ease: "easeInOut" },
              marginTop: { duration: 0.28, ease: "easeInOut" },
              opacity: { duration: 0.2, ease: "easeOut" },
              y: { duration: 0.24, ease: "easeOut" },
              filter: { duration: 0.24, ease: "easeOut" },
              boxShadow: { duration: 1.35, ease: "easeInOut", times: [0, 0.12, 0.24, 0.38, 0.52, 0.68, 1] },
            }}
            className="overflow-hidden"
          >
            <div className={`rounded-xl border px-3 py-2 text-sm backdrop-blur-md ${flashToneClass(flashMessage.tone)}`}>
              <span className="inline-flex items-center gap-2">
                <FlashIcon tone={flashMessage.tone} />
                {flashMessage.message}
              </span>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <motion.section
        key={location.pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut" }}
        className="mt-4"
      >
        {children}
      </motion.section>
    </div>
  );
}
