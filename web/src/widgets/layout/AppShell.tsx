import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Home,
  Info,
  LogOut,
  ShieldAlert,
  Trophy,
  User as UserIcon,
} from "lucide-react";
import { useAuth } from "@/app/providers/useAuth";
import { InvitationList } from "@/features/matches/InvitationList";
import { useInvitations } from "@/features/matches/useInvitations";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";
import { cn } from "@/shared/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Inicio", icon: Home },
  { to: "/ranking", label: "Ranking", icon: Trophy },
  { to: "/profile", label: "Perfil", icon: UserIcon },
];
const MATCHMAKING_MATCH_KEY = "ataxx.matchmaking.match.v1";
const INVITE_SFX_PATH = "/sfx/queue_found.ogg";

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

function playSfx(path: string, volume = 0.24): void {
  try {
    const audio = new Audio(path);
    audio.volume = Math.max(0, Math.min(1, volume));
    const playResult = audio.play();
    if (playResult && typeof playResult.catch === "function") {
      void playResult.catch(() => {});
    }
  } catch {
    // ignore browser/runtime audio errors
  }
}

type AppShellProps = {
  children: React.ReactNode;
  onNavigateAttempt?: (to: string) => boolean;
  onLogoutAttempt?: () => boolean;
};

export function AppShell({ children, onNavigateAttempt, onLogoutAttempt }: AppShellProps): JSX.Element {
  const { isAuthenticated, user, accessToken, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [flashMessage, setFlashMessage] = useState<FlashMessage | null>(null);
  const [invitePanelOpen, setInvitePanelOpen] = useState(false);
  const invitePanelRef = useRef<HTMLDivElement | null>(null);
  const previousInviteCountRef = useRef(0);
  const {
    invitations: pendingInvitations,
    actionLoadingId: inviteActionLoadingId,
    acceptInvitationById,
    rejectInvitationById,
  } = useInvitations({
    accessToken,
    enabled: isAuthenticated,
    includeInitialFetch: true,
    scope: "appshell",
    fallbackPollingMs: 3000,
  });
  const incomingInvitations = pendingInvitations.length;
  const sortedInvitations = useMemo(
    () =>
      [...pendingInvitations].sort((a, b) => {
        const aTs = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bTs = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bTs - aTs;
      }),
    [pendingInvitations],
  );

  useEffect(() => {
    const previous = previousInviteCountRef.current;
    if (incomingInvitations > previous) {
      const incomingDelta = incomingInvitations - previous;
      playSfx(INVITE_SFX_PATH, 0.24);
      setFlashMessage({
        message:
          incomingDelta > 1
            ? `Recibiste ${incomingDelta} invitaciones 1v1 nuevas.`
            : "Recibiste una invitacion 1v1.",
        tone: "info",
      });
    }
    previousInviteCountRef.current = incomingInvitations;
  }, [incomingInvitations]);

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

  useEffect(() => {
    if (pendingInvitations.length === 0) {
      setInvitePanelOpen(false);
    }
  }, [pendingInvitations.length]);

  useEffect(() => {
    if (!invitePanelOpen) {
      return;
    }
    const onPointerDown = (event: MouseEvent): void => {
      const panel = invitePanelRef.current;
      if (panel === null) {
        return;
      }
      if (event.target instanceof Node && !panel.contains(event.target)) {
        setInvitePanelOpen(false);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [invitePanelOpen]);

  const acceptPendingInvitation = async (gameId: string): Promise<void> => {
    try {
      await acceptInvitationById(gameId);
      sessionStorage.setItem(
        MATCHMAKING_MATCH_KEY,
        JSON.stringify({
          gameId,
          matchedWith: "human",
          createdAt: Date.now(),
          source: "invite",
        }),
      );
      setInvitePanelOpen(false);
      navigate("/match?queue=1", {
        state: { flash: { message: "Invitacion aceptada. Entrando a la arena...", tone: "success" } },
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : "No se pudo aceptar la invitacion.";
      setFlashMessage({ message: detail, tone: "error" });
    }
  };

  const rejectPendingInvitation = async (gameId: string): Promise<void> => {
    try {
      await rejectInvitationById(gameId);
      setFlashMessage({ message: "Invitacion rechazada.", tone: "info" });
    } catch (error) {
      const detail = error instanceof Error ? error.message : "No se pudo rechazar la invitacion.";
      setFlashMessage({ message: detail, tone: "error" });
    }
  };

  return (
    <div className="mx-auto min-h-screen w-full max-w-[1320px] px-3 py-3 sm:px-5">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
        className="relative overflow-visible rounded-2xl border border-zinc-800/90 bg-zinc-950/60 px-4 py-3 backdrop-blur-xl"
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

        <nav className="mt-3 flex items-center gap-1 overflow-visible border-t border-zinc-800/80 pt-2">
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
                  {item.to === "/profile" && incomingInvitations > 0 ? (
                    <span
                      aria-label={`${incomingInvitations} invitaciones pendientes`}
                      className="inline-flex min-w-[1.2rem] items-center justify-center rounded-full border border-lime-300/70 bg-lime-300/18 px-1 text-[10px] font-semibold leading-4 text-lime-200 shadow-[0_0_10px_rgba(163,230,53,0.3)]"
                    >
                      {incomingInvitations > 9 ? "9+" : incomingInvitations}
                    </span>
                  ) : null}
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
          {isAuthenticated ? (
            <div className="relative ml-auto" ref={invitePanelRef}>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className={cn(
                  "relative rounded-md px-3 text-textDim transition-all duration-200 hover:-translate-y-0.5 hover:text-lime-200",
                  incomingInvitations > 0 &&
                    "border border-lime-300/55 bg-lime-300/12 text-lime-200 shadow-[0_0_18px_rgba(163,230,53,0.28)]",
                  invitePanelOpen && "bg-lime-400/10 text-lime-200 shadow-[0_0_12px_rgba(163,230,53,0.12)]",
                )}
                aria-label="Ver invitaciones"
                onClick={() => setInvitePanelOpen((prev) => !prev)}
              >
                <motion.span
                  animate={
                    incomingInvitations > 0
                      ? { rotate: [0, -10, 8, -6, 0], scale: [1, 1.08, 1] }
                      : { rotate: 0, scale: 1 }
                  }
                  transition={{
                    duration: incomingInvitations > 0 ? 1.2 : 0.2,
                    repeat: incomingInvitations > 0 ? Infinity : 0,
                    repeatDelay: 1.8,
                    ease: "easeInOut",
                  }}
                  className="inline-flex"
                >
                  <Bell className="h-3.5 w-3.5" />
                </motion.span>
                {incomingInvitations > 0 ? (
                  <span className="ml-1.5 inline-flex min-w-[1.2rem] items-center justify-center rounded-full border border-lime-300/80 bg-lime-300/24 px-1 text-[10px] font-semibold leading-4 text-lime-100 shadow-[0_0_14px_rgba(163,230,53,0.42)]">
                    {incomingInvitations > 9 ? "9+" : incomingInvitations}
                  </span>
                ) : null}
              </Button>
              <AnimatePresence>
                {invitePanelOpen ? (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.98 }}
                    transition={{ duration: 0.18, ease: "easeOut" }}
                    className="absolute right-0 top-11 z-40 w-80 rounded-lg border border-zinc-700/90 bg-zinc-950/95 p-2 shadow-[0_18px_44px_rgba(0,0,0,0.55)] backdrop-blur"
                  >
                    <div className="mb-2 flex items-center justify-between px-1">
                      <p className="text-xs uppercase tracking-[0.12em] text-zinc-400">Invitaciones</p>
                      <span className="text-xs font-semibold text-lime-300">{incomingInvitations}</span>
                    </div>
                    <div className="max-h-72 space-y-1.5 overflow-y-auto pr-1">
                      <InvitationList
                        items={sortedInvitations}
                        isLoading={false}
                        isError={false}
                        actionLoadingId={inviteActionLoadingId}
                        variant="panel"
                        emptyText="Sin invitaciones pendientes."
                        onAccept={(gameId) => {
                          void acceptPendingInvitation(gameId);
                        }}
                        onReject={(gameId) => {
                          void rejectPendingInvitation(gameId);
                        }}
                      />
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>
          ) : null}
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
