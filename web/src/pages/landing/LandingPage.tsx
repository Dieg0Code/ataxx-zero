import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Check, Clock3, Crown, Search, ShieldCheck, Trophy, Users, X } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/app/providers/useAuth";
import {
  acceptMatchedQueue,
  joinRankedQueue,
  leaveQueue,
  openQueueSocket,
  rejectMatchedQueue,
  type MatchedWith,
  type QueueWsEvent,
} from "@/features/matchmaking/api";
import { fetchPersistedGameSummary } from "@/features/match/persistence";
import { fetchPublicPlayers, prefetchPublicPlayers } from "@/features/identity/api";
import { AppShell } from "@/widgets/layout/AppShell";
import { fetchActiveSeason, fetchLeaderboard } from "@/features/ranking/api";
import { assetUrl } from "@/shared/lib/assets";
import { playSfx, primeSfx, primeSfxOnFirstInteraction } from "@/shared/lib/sfx";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Button } from "@/shared/ui/button";
import { Badge } from "@/shared/ui/badge";

const HOME_TOP_LIMIT = 3;
const MATCHMAKING_MATCH_KEY = "ataxx.matchmaking.match.v1";
const MATCHMAKING_AUTOQUEUE_KEY = "ataxx.matchmaking.autoqueue.v1";
const MATCHMAKING_PRESET_KEY = "ataxx.matchmaking.preset.v1";
const GUEST_PROFILE_KEY = "ataxx.guest.profile.v1";
const GUEST_QUEUE_DELAY_MS = 1200;
const GUEST_BOT_FALLBACK = [
  "kernelseraph",
  "cipherpraetor",
  "aetherglyph",
  "voidkernel",
  "neonwarden",
] as const;
const QUEUE_SFX = {
  search: assetUrl("sfx/start.ogg"),
  found: assetUrl("sfx/queue_found.ogg"),
  accept: assetUrl("sfx/queue_accept.ogg"),
  reject: assetUrl("sfx/queue_reject.ogg"),
} as const;

type QueueMatch = {
  gameId: string;
  matchedWith: MatchedWith;
  opponentUsername: string | null;
  createdAt: number;
  source: "queue";
};
type QueuePhase = "idle" | "searching" | "confirming" | "loading";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function normalizeGuestUsername(value: string): string {
  return value
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_-]/g, "")
    .slice(0, 18);
}

function fallbackOpponentName(matchedWith: MatchedWith): string {
  return matchedWith === "human" ? "rival humano" : "bot rival";
}

function saveMatchedGame(gameId: string, matchedWith: MatchedWith): void {
  const payload: QueueMatch = {
    gameId,
    matchedWith,
    createdAt: Date.now(),
    source: "queue",
  };
  sessionStorage.setItem(MATCHMAKING_MATCH_KEY, JSON.stringify(payload));
}

export function LandingPage(): JSX.Element {
  const { user, isAuthenticated, accessToken } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [queueActive, setQueueActive] = useState(false);
  const [queueSeconds, setQueueSeconds] = useState(0);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [queueJoining, setQueueJoining] = useState(false);
  const [ctaReward, setCtaReward] = useState(false);
  const [pendingMatch, setPendingMatch] = useState<QueueMatch | null>(null);
  const [acceptingMatch, setAcceptingMatch] = useState(false);
  const [rejectingMatch, setRejectingMatch] = useState(false);
  const [acceptCountdown, setAcceptCountdown] = useState(12);
  const [pendingOpponentName, setPendingOpponentName] = useState<string | null>(null);
  const [guestModalOpen, setGuestModalOpen] = useState(false);
  const [guestUsername, setGuestUsername] = useState(() => {
    try {
      const raw = localStorage.getItem(GUEST_PROFILE_KEY);
      if (raw === null) {
        return "";
      }
      const parsed = JSON.parse(raw) as { username?: string };
      return typeof parsed.username === "string" ? parsed.username : "";
    } catch {
      return "";
    }
  });
  const [guestQueueing, setGuestQueueing] = useState(false);
  const joinRequestSeqRef = useRef(0);
  const guestQueueCancelledRef = useRef(false);

  useEffect(() => {
    const sfxPaths = Object.values(QUEUE_SFX);
    primeSfx(sfxPaths, 4);
    primeSfxOnFirstInteraction(sfxPaths, 4);
  }, []);

  const seasonQuery = useQuery({
    queryKey: ["activeSeason"],
    queryFn: fetchActiveSeason,
    staleTime: 60_000,
  });

  const leaderboardQuery = useQuery({
    queryKey: ["home-top-leaderboard", seasonQuery.data?.id],
    queryFn: () => fetchLeaderboard(seasonQuery.data!.id, HOME_TOP_LIMIT, 0),
    enabled: Boolean(seasonQuery.data?.id),
  });

  const topPlayers = useMemo(() => leaderboardQuery.data?.items ?? [], [leaderboardQuery.data?.items]);
  const queuePhase = useMemo<QueuePhase>(() => {
    if (!queueActive) {
      return "idle";
    }
    if (queueSeconds < 4) {
      return "searching";
    }
    if (queueSeconds < 8) {
      return "confirming";
    }
    return "loading";
  }, [queueActive, queueSeconds]);

  const queuePhaseText =
    queuePhase === "searching"
      ? "Buscando rival..."
      : queuePhase === "confirming"
        ? "Confirmando partida..."
        : queuePhase === "loading"
          ? "Cargando tablero..."
          : "En espera";

  const queuePhaseBadge =
    queuePhase === "searching"
      ? ("success" as const)
      : queuePhase === "confirming"
        ? ("warning" as const)
        : queuePhase === "loading"
          ? ("default" as const)
          : ("default" as const);
  const queuePulseClass =
    queuePhase === "searching"
      ? "border-lime-300/70 border-t-lime-200"
      : queuePhase === "confirming"
        ? "border-amber/70 border-t-amber"
        : "border-primary/70 border-t-primary";
  const queueGlowClass =
    queuePhase === "loading" ? "shadow-[0_0_16px_rgba(163,230,53,0.45)]" : "shadow-[0_0_8px_rgba(163,230,53,0.18)]";
  const acceptProgress = Math.max(0, Math.min(1, acceptCountdown / 12));
  const queueCtaCancelMode = queueActive || queueJoining;

  const emitFlash = useCallback(
    (message: string, tone: "success" | "warning" | "error" | "info"): void => {
      navigate(`${location.pathname}${location.search}${location.hash}`, {
        replace: true,
        state: { flash: { message, tone } },
      });
    },
    [location.hash, location.pathname, location.search, navigate],
  );

  useEffect(() => {
    if (!queueActive) {
      return;
    }
    const interval = window.setInterval(() => {
      setQueueSeconds((prev) => prev + 1);
    }, 1000);
    return () => window.clearInterval(interval);
  }, [queueActive]);

  const openMatchAccept = (
    gameId: string,
    matchedWith: MatchedWith,
    opponentUsername: string | null = null,
    source: "queue" = "queue",
  ): void => {
    playSfx(QUEUE_SFX.found, 0.24);
    setQueueActive(false);
    setQueueSeconds(0);
    setQueueJoining(false);
    setQueueError(null);
    setAcceptCountdown(12);
    setPendingMatch({
      gameId,
      matchedWith,
      opponentUsername,
      createdAt: Date.now(),
      source,
    });
    const normalizedOpponentUsername = opponentUsername?.trim();
    setPendingOpponentName(
      normalizedOpponentUsername && normalizedOpponentUsername.length > 0
        ? normalizedOpponentUsername
        : fallbackOpponentName(matchedWith),
    );
  };

  useEffect(() => {
    if (pendingMatch === null) {
      setPendingOpponentName(null);
      return;
    }
    if (pendingMatch.opponentUsername !== null && pendingMatch.opponentUsername.trim().length > 0) {
      setPendingOpponentName(pendingMatch.opponentUsername);
      return;
    }
    if (accessToken === null) {
      setPendingOpponentName(fallbackOpponentName(pendingMatch.matchedWith));
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const game = await fetchPersistedGameSummary(accessToken, pendingMatch.gameId);
        const summaryName =
          game.player1_id === user?.id
            ? game.player2_username
            : game.player2_id === user?.id
              ? game.player1_username
              : game.player2_username ?? game.player1_username;
        if (typeof summaryName === "string" && summaryName.trim().length > 0) {
          if (!cancelled) {
            setPendingOpponentName(summaryName);
          }
          return;
        }
        const opponentUserId = game.player2_id;
        if (opponentUserId === null) {
          if (!cancelled) {
            setPendingOpponentName(fallbackOpponentName(pendingMatch.matchedWith));
          }
          return;
        }
        const players = await fetchPublicPlayers(accessToken, { limit: 200 });
        const opponent = players.find((player) => player.user_id === opponentUserId);
        if (!cancelled) {
          setPendingOpponentName(opponent?.username ?? fallbackOpponentName(pendingMatch.matchedWith));
        }
      } catch {
        if (!cancelled) {
          setPendingOpponentName(fallbackOpponentName(pendingMatch.matchedWith));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken, pendingMatch, user?.id]);

  useEffect(() => {
    if (pendingMatch === null) {
      return;
    }
    if (acceptingMatch || rejectingMatch) {
      return;
    }
    const timer = window.setInterval(() => {
      setAcceptCountdown((prev) => {
        if (prev <= 1) {
          window.clearInterval(timer);
          void (async () => {
            if (accessToken !== null) {
              try {
                await rejectMatchedQueue(accessToken);
              } catch {
                // Best effort server reject.
              }
            }
            setPendingMatch(null);
            emitFlash("Emparejamiento expirado. Vuelve a buscar partida.", "warning");
          })();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [acceptingMatch, accessToken, emitFlash, pendingMatch, rejectingMatch]);

  useEffect(() => {
    if (!queueActive || accessToken === null) {
      return;
    }

    const onEvent = (event: QueueWsEvent): void => {
      if (event.type !== "queue.status") {
        return;
      }
      const status = event.payload;
      if (status.status === "matched" && status.game_id !== null && status.matched_with !== null) {
        openMatchAccept(status.game_id, status.matched_with, status.opponent_username);
      }
    };

    const socket = openQueueSocket(accessToken, onEvent);
    socket.onerror = () => {
      setQueueError("Conexion en tiempo real interrumpida. Vuelve a intentar cola.");
      setQueueActive(false);
    };

    return () => {
      socket.close();
    };
  }, [accessToken, navigate, queueActive]);

  const startQueue = useCallback(async (): Promise<void> => {
    if (queueActive || queueJoining || pendingMatch !== null) {
      return;
    }
    if (!isAuthenticated || accessToken === null) {
      setGuestModalOpen(true);
      return;
    }
    setQueueJoining(true);
    joinRequestSeqRef.current += 1;
    const requestSeq = joinRequestSeqRef.current;
    setCtaReward(true);
    setQueueError(null);
    playSfx(QUEUE_SFX.search, 0.22);
    try {
      const joined = await joinRankedQueue(accessToken);
      if (requestSeq !== joinRequestSeqRef.current) {
        try {
          await leaveQueue(accessToken);
        } catch {
          // best effort rollback when join was canceled locally.
        }
        return;
      }
      if (joined.status === "matched" && joined.game_id !== null && joined.matched_with !== null) {
        openMatchAccept(joined.game_id, joined.matched_with, joined.opponent_username);
        return;
      }
      setQueueActive(true);
      setQueueSeconds(0);
      emitFlash("Entraste a la cola competitiva.", "info");
    } catch (error) {
      const message = error instanceof Error ? error.message : "No se pudo entrar a cola.";
      setQueueError(message);
      setQueueActive(false);
      emitFlash("No se pudo entrar a cola.", "error");
    } finally {
      setQueueJoining(false);
    }
  }, [accessToken, emitFlash, isAuthenticated, pendingMatch, queueActive, queueJoining]);

  const startGuestQueue = useCallback(async (): Promise<void> => {
    if (guestQueueing) {
      return;
    }
    const normalized = normalizeGuestUsername(guestUsername);
    if (normalized.length < 3) {
      setQueueError("El nombre invitado debe tener al menos 3 caracteres validos.");
      return;
    }
    setGuestQueueing(true);
    guestQueueCancelledRef.current = false;
    setGuestModalOpen(false);
    setQueueError(null);
    setQueueActive(true);
    setQueueSeconds(0);
    playSfx(QUEUE_SFX.search, 0.22);

    try {
      localStorage.setItem(GUEST_PROFILE_KEY, JSON.stringify({ username: normalized }));
      await sleep(GUEST_QUEUE_DELAY_MS);
      if (guestQueueCancelledRef.current) {
        return;
      }
      const rankedBotNames = topPlayers
        .filter((entry) => entry.is_bot)
        .map((entry) => (entry.username ?? "").trim())
        .filter((name) => name.length > 0 && name.toLowerCase() !== normalized.toLowerCase());
      const rivalPool = rankedBotNames.length > 0 ? rankedBotNames : [...GUEST_BOT_FALLBACK];
      const rivalName = rivalPool[Math.floor(Math.random() * rivalPool.length)] ?? "rival";
      const heuristicLevels = ["easy", "normal", "hard", "apex", "gambit", "sentinel"] as const;
      const level = heuristicLevels[Math.floor(Math.random() * heuristicLevels.length)] ?? "normal";
      sessionStorage.setItem(
        MATCHMAKING_PRESET_KEY,
        JSON.stringify({
          opponentProfile: "heuristic",
          heuristicLevel: level,
          guestMode: true,
          guestUsername: normalized,
          guestRivalName: rivalName,
          createdAt: Date.now(),
        }),
      );
      navigate("/match?queue=guest", {
        state: {
          flash: { message: "Cola casual lista. Rival encontrado.", tone: "success" },
        },
      });
    } finally {
      setQueueActive(false);
      setQueueSeconds(0);
      setGuestQueueing(false);
    }
  }, [guestQueueing, guestUsername, navigate, topPlayers]);

  useEffect(() => {
    if (!ctaReward) {
      return;
    }
    const timer = window.setTimeout(() => {
      setCtaReward(false);
    }, 900);
    return () => window.clearTimeout(timer);
  }, [ctaReward]);

  const cancelQueue = useCallback((): void => {
    joinRequestSeqRef.current += 1;
    guestQueueCancelledRef.current = true;
    // Apply local cancel immediately; backend leave can finish in background.
    setQueueActive(false);
    setQueueJoining(false);
    setQueueSeconds(0);
    setQueueError(null);
    playSfx(QUEUE_SFX.reject, 0.2);
    emitFlash("Cola cancelada.", "warning");
    if (accessToken !== null) {
      void leaveQueue(accessToken).catch(() => {
        // Ignore backend leave errors and keep local state canceled.
      });
    }
  }, [accessToken, emitFlash]);

  useEffect(() => {
    if (!isAuthenticated || accessToken === null) {
      return;
    }
    const raw = sessionStorage.getItem(MATCHMAKING_AUTOQUEUE_KEY);
    if (raw === null) {
      return;
    }
    sessionStorage.removeItem(MATCHMAKING_AUTOQUEUE_KEY);
    void startQueue();
  }, [accessToken, isAuthenticated, startQueue]);

  const acceptMatch = async (): Promise<void> => {
    if (pendingMatch === null) {
      return;
    }
    setAcceptingMatch(true);
    playSfx(QUEUE_SFX.accept, 0.24);
    try {
      if (accessToken !== null) {
        const decision = await acceptMatchedQueue(accessToken);
        if (decision.game_id !== null) {
          saveMatchedGame(decision.game_id, pendingMatch.matchedWith);
        } else {
          saveMatchedGame(pendingMatch.gameId, pendingMatch.matchedWith);
        }
      } else {
        saveMatchedGame(pendingMatch.gameId, pendingMatch.matchedWith);
      }
      setPendingMatch(null);
      navigate("/match?queue=1", {
        state: { flash: { message: "Partida aceptada. Entrando a la arena...", tone: "success" } },
      });
    } finally {
      setAcceptingMatch(false);
    }
  };

  const rejectMatch = async (): Promise<void> => {
    if (pendingMatch === null) {
      return;
    }
    setRejectingMatch(true);
    playSfx(QUEUE_SFX.reject, 0.2);
    try {
      if (accessToken !== null) {
        try {
          await rejectMatchedQueue(accessToken);
        } catch {
          // Best effort server reject.
        }
      }
      setPendingMatch(null);
      emitFlash("Partida rechazada. Volviste al lobby.", "info");
    } finally {
      setRejectingMatch(false);
    }
  };

  const prefetchMatchSetup = useCallback((): void => {
    if (!isAuthenticated || accessToken === null) {
      return;
    }
    void prefetchPublicPlayers(accessToken, { limit: 200 });
  }, [accessToken, isAuthenticated]);

  return (
    <AppShell>
      <AnimatePresence>
        {pendingMatch !== null ? (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 px-4 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="relative w-full max-w-md overflow-hidden rounded-xl border border-lime-300/45 bg-zinc-950/95 p-5 shadow-[0_0_44px_rgba(163,230,53,0.22)]"
              initial={{ opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <motion.div
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 opacity-40"
                style={{
                  background:
                    "linear-gradient(120deg, rgba(163,230,53,0.08) 0%, rgba(163,230,53,0.02) 28%, transparent 52%)",
                }}
                animate={{ opacity: [0.3, 0.5, 0.3] }}
                transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
              />
              <motion.div
                aria-hidden="true"
                className="pointer-events-none absolute left-0 right-0 top-0 h-px bg-lime-300/80"
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
              />
              <button
                type="button"
                aria-label="Cerrar modal"
                className="absolute right-4 top-4 rounded-md border border-zinc-600/70 bg-zinc-900/70 p-1 text-zinc-300 transition hover:border-red-400/50 hover:text-red-200"
                disabled={acceptingMatch || rejectingMatch}
                onClick={() => {
                  void rejectMatch();
                }}
              >
                <X className="h-4 w-4" />
              </button>
              <p className="text-xs uppercase tracking-[0.14em] text-lime-300">Partida encontrada</p>
              <h3 className="mt-2 text-xl font-semibold text-zinc-100">Aceptar enfrentamiento</h3>
              <p className="mt-1 text-xs uppercase tracking-[0.12em] text-zinc-500">
                Emparejamiento de cola
              </p>
              <p className="mt-3 text-sm text-zinc-300">
                Rival:{" "}
                <span className="font-semibold text-zinc-100">
                  {pendingOpponentName ?? fallbackOpponentName(pendingMatch.matchedWith)}
                </span>
              </p>
              <p className="mt-1 text-sm text-zinc-400">
                Tiempo restante: <span className="font-semibold text-lime-300">{acceptCountdown}s</span>
              </p>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full border border-lime-300/35 bg-zinc-900/90">
                <motion.div
                  className="h-full bg-[linear-gradient(90deg,rgba(163,230,53,0.55),rgba(190,242,100,0.95))]"
                  animate={{ width: `${Math.round(acceptProgress * 100)}%` }}
                  transition={{ duration: 0.2, ease: "easeOut" }}
                />
              </div>
              <div className="mt-4 grid grid-cols-1 gap-2">
                <Button
                  type="button"
                  variant="default"
                  disabled={acceptingMatch || rejectingMatch}
                  onClick={() => {
                    void acceptMatch();
                  }}
                  className="border border-lime-300/65 bg-lime-300 text-black hover:bg-lime-200"
                >
                  <Check className="mr-1.5 h-4 w-4" />
                    {acceptingMatch ? "Aceptando..." : "Aceptar"}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {guestModalOpen ? (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="relative w-full max-w-md rounded-xl border border-lime-300/45 bg-zinc-950/95 p-5 shadow-[0_0_40px_rgba(163,230,53,0.2)]"
              initial={{ opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <button
                type="button"
                aria-label="Cerrar modal invitado"
                className="absolute right-4 top-4 rounded-md border border-zinc-600/70 bg-zinc-900/70 p-1 text-zinc-300 transition hover:border-red-400/50 hover:text-red-200"
                onClick={() => setGuestModalOpen(false)}
              >
                <X className="h-4 w-4" />
              </button>
              <p className="text-xs uppercase tracking-[0.14em] text-lime-300">Modo invitado</p>
              <h3 className="mt-2 text-xl font-semibold text-zinc-100">Entrar a cola casual</h3>
              <p className="mt-2 text-sm text-zinc-300">
                Escribe un username rapido. Esta partida no guarda replay ni puntos.
              </p>
              <label className="mt-4 block text-xs uppercase tracking-[0.12em] text-zinc-400" htmlFor="guest-username">
                Username
              </label>
              <input
                id="guest-username"
                type="text"
                value={guestUsername}
                onChange={(event) => setGuestUsername(event.target.value)}
                placeholder="ej: neonrunner"
                className="mt-1 h-10 w-full rounded-md border border-zinc-700 bg-zinc-950/80 px-3 text-sm text-zinc-100 outline-none transition focus:border-lime-300/60"
                maxLength={24}
              />
              <div className="mt-4 grid grid-cols-1 gap-2">
                <Button
                  type="button"
                  variant="default"
                  disabled={guestQueueing}
                  onClick={() => {
                    void startGuestQueue();
                  }}
                  className="border border-lime-300/65 bg-lime-300 text-black hover:bg-lime-200"
                >
                  {guestQueueing ? "Entrando..." : "Entrar como invitado"}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
          <Card className="border-zinc-800/90 bg-[linear-gradient(180deg,rgba(132,204,22,0.04),rgba(0,0,0,0.38))] shadow-[0_0_16px_rgba(132,204,22,0.08)]">
            <CardHeader>
              <Badge variant="default" className="w-fit border-lime-300/45 bg-lime-300/10 text-lime-200">
                UNDERBYTELABS // ATAXX-ZERO
              </Badge>
              <CardTitle className="mt-3 text-3xl leading-tight sm:text-4xl">Arena Ataxx</CardTitle>
              <CardDescription className="max-w-2xl">
                Duelo tactico en tiempo real. Entra a cola, encuentra rival y juega.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <motion.div
                  className="relative"
                  animate={
                    queueCtaCancelMode
                      ? {}
                      : {
                          scale: [1, 1.018, 1, 1.01, 1],
                          y: [0, -0.35, 0, -0.2, 0],
                        }
                  }
                  transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
                >
                  <AnimatePresence>
                    {ctaReward ? (
                      <motion.span
                        key="cta-reward-ring"
                        aria-hidden="true"
                        className="pointer-events-none absolute inset-0 rounded-md border border-lime-200/75"
                        initial={{ opacity: 0.8, scale: 0.94 }}
                        animate={{ opacity: 0, scale: 1.16 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.85, ease: "easeOut" }}
                      />
                    ) : null}
                  </AnimatePresence>
                  {!queueActive && !queueJoining ? (
                    <motion.span
                      aria-hidden="true"
                      className="pointer-events-none absolute inset-0 rounded-md bg-lime-300/25 blur-[12px]"
                      animate={{ opacity: [0.26, 0.46, 0.26] }}
                      transition={{ duration: 1.7, repeat: Infinity, ease: "easeInOut" }}
                    />
                  ) : null}
                  <Button
                    type="button"
                    variant="default"
                    className={`group relative z-[1] overflow-hidden border transition-all duration-200 ${
                      queueCtaCancelMode
                        ? "border-red-300/60 bg-red-400/18 text-red-100 shadow-[0_0_22px_rgba(248,113,113,0.28)] hover:bg-red-400/26"
                        : "border-lime-200/70 bg-lime-300 text-black shadow-[0_0_20px_rgba(163,230,53,0.4)] hover:bg-lime-200 hover:shadow-[0_0_28px_rgba(163,230,53,0.58)]"
                    }`}
                    onClick={() => {
                      if (queueCtaCancelMode) {
                        cancelQueue();
                        return;
                      }
                      void startQueue();
                    }}
                    disabled={false}
                    aria-busy={queueJoining}
                    aria-label={queueCtaCancelMode ? "Detener busqueda" : "Buscar partida"}
                  >
                    {!queueActive && !queueJoining ? (
                      <motion.span
                        aria-hidden="true"
                        className="pointer-events-none absolute -left-8 top-0 h-full w-6 rotate-12 bg-white/35 blur-[1px]"
                        initial={{ x: -24, opacity: 0 }}
                        animate={{
                          x: [0, 180, 180],
                          opacity: [0, 0.22, 0],
                        }}
                        transition={{
                          duration: 3.8,
                          repeat: Infinity,
                          repeatDelay: 1.6,
                          ease: "easeInOut",
                        }}
                      />
                    ) : null}
                    <span className="mr-2 inline-flex h-4 w-4 items-center justify-center overflow-hidden">
                      <AnimatePresence mode="wait" initial={false}>
                        {queueCtaCancelMode ? (
                          <motion.span
                            key="queue-cta-icon-cancel"
                            className="inline-flex"
                            initial={{ opacity: 0, scale: 0.78, rotate: -70 }}
                            animate={{ opacity: 1, scale: 1, rotate: 0 }}
                            exit={{ opacity: 0, scale: 0.78, rotate: 70 }}
                            transition={{ duration: 0.14, ease: "easeOut" }}
                          >
                            <X className="h-4 w-4" />
                          </motion.span>
                        ) : (
                          <motion.span
                            key="queue-cta-icon-search"
                            className="inline-flex"
                            initial={{ opacity: 0, scale: 0.86 }}
                            animate={{ opacity: 1, scale: [1, 1.08, 1] }}
                            exit={{ opacity: 0, scale: 0.86 }}
                            transition={{
                              opacity: { duration: 0.14, ease: "easeOut" },
                              scale: { duration: 2.2, repeat: Infinity, ease: "easeInOut" },
                            }}
                          >
                            <Search className="h-4 w-4" />
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </span>
                    <span className="inline-flex min-w-[8.5rem] justify-center text-center">
                      <AnimatePresence mode="wait" initial={false}>
                        {queueCtaCancelMode ? (
                          <motion.span
                            key="queue-cta-label-cancel"
                            initial={{ opacity: 0, y: 3 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -3 }}
                            transition={{ duration: 0.15, ease: "easeOut" }}
                          >
                            Detener busqueda
                          </motion.span>
                        ) : (
                          <motion.span
                            key="queue-cta-label-search"
                            initial={{ opacity: 0, y: 3 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -3 }}
                            transition={{ duration: 0.15, ease: "easeOut" }}
                          >
                            Buscar partida
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </span>
                  </Button>
                </motion.div>
                <Button
                  asChild
                  variant="secondary"
                  size="sm"
                  className="h-9 border border-lime-300/45 bg-lime-300/10 px-2 text-lime-100 hover:bg-lime-300/18"
                >
                  <Link to="/lobby" onMouseEnter={prefetchMatchSetup} onFocus={prefetchMatchSetup} onTouchStart={prefetchMatchSetup}>
                    <Users className="mr-2 h-4 w-4" />
                    Crear sala
                  </Link>
                </Button>
                <Button
                  asChild
                  variant="ghost"
                  size="sm"
                  className="h-8 border border-zinc-800/70 bg-transparent px-2 text-zinc-500 hover:bg-zinc-900/60 hover:text-zinc-200"
                >
                  <Link
                    to="/profile#laboratorio"
                    onMouseEnter={prefetchMatchSetup}
                    onFocus={prefetchMatchSetup}
                    onTouchStart={prefetchMatchSetup}
                  >
                    Configuracion avanzada
                    <ArrowRight className="ml-2 h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-1" />
                  </Link>
                </Button>
              </div>

              <div className="rounded-lg border border-zinc-700/80 bg-zinc-950/60 p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="inline-flex items-center gap-2 text-sm text-textMain">
                    <Clock3 className="h-4 w-4 text-lime-300" />
                    Cola competitiva
                  </p>
                  {queueActive ? (
                    <Badge variant={queuePhaseBadge}>{queuePhaseText}</Badge>
                  ) : (
                    <Badge variant="default">Inactiva</Badge>
                  )}
                </div>

                {queueActive ? (
                  <div className="mt-3 space-y-2">
                    <div className="flex items-center justify-between rounded-md border border-zinc-700/70 bg-zinc-950/80 px-3 py-2">
                      <div className="inline-flex items-center gap-2">
                        <motion.div
                          className={`h-4 w-4 rounded-full border-2 ${queuePulseClass} ${queueGlowClass}`}
                          animate={{
                            rotate: 360,
                            scale: queuePhase === "loading" ? [1, 1.08, 1] : [1, 1.03, 1],
                          }}
                          transition={{
                            rotate: { duration: 0.9, ease: "linear", repeat: Infinity },
                            scale: { duration: queuePhase === "loading" ? 0.9 : 1.2, repeat: Infinity, ease: "easeInOut" },
                          }}
                        />
                        <p className="text-sm text-textDim">Tiempo en cola: {queueSeconds}s</p>
                      </div>
                      <p className="text-xs text-textDim">{queuePhaseText}</p>
                    </div>
                    <div className="flex items-center justify-between">
                      <motion.p
                        className="text-xs text-textDim"
                        animate={{ opacity: [0.45, 1, 0.45] }}
                        transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                      >
                        {queuePhaseText}
                      </motion.p>
                    </div>
                  </div>
                ) : (
                  <p className="mt-2 text-sm text-textDim">Pulsa "Buscar partida" para entrar a la cola automatica.</p>
                )}
                {queueError ? <p className="mt-2 text-xs text-redGlow">{queueError}</p> : null}
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-lg border border-zinc-700/80 bg-zinc-950/60 px-3 py-2">
                  <p className="text-lg font-semibold text-textMain">7x7</p>
                  <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">tablero</p>
                </div>
                <div className="rounded-lg border border-zinc-700/80 bg-zinc-950/60 px-3 py-2">
                  <p className="text-lg font-semibold text-textMain">ranked</p>
                  <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">cola</p>
                </div>
                <div className="rounded-lg border border-zinc-700/80 bg-zinc-950/60 px-3 py-2">
                  <p className="text-lg font-semibold text-textMain">live</p>
                  <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">replay</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <div className="grid gap-4">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, delay: 0.05 }}>
            <Card className="border-zinc-800/90 bg-zinc-950/60">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <ShieldCheck className="h-4 w-4 text-lime-300" />
                  Sesion activa
                </CardTitle>
                <CardDescription>Tu progreso se refleja en ranking y historial de partidas.</CardDescription>
              </CardHeader>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2, delay: 0.1 }}>
            <Card className="border-zinc-800/90 bg-zinc-950/60">
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Trophy className="h-4 w-4 text-lime-300" />
                    Temporada en vivo
                  </CardTitle>
                  {seasonQuery.data?.name ? <Badge variant="success">{seasonQuery.data.name}</Badge> : null}
                </div>
                <CardDescription>Top actual de la temporada.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {seasonQuery.isLoading ? <p className="text-sm text-textDim">Cargando temporada...</p> : null}
                {seasonQuery.isError ? <p className="text-sm text-redGlow">No hay temporada activa disponible.</p> : null}
                {!seasonQuery.isLoading && !seasonQuery.isError && topPlayers.length === 0 ? (
                  <p className="text-sm text-textDim">Sin datos de ranking por ahora.</p>
                ) : null}
                {topPlayers.map((entry) => {
                  const isLeader = entry.rank === 1;
                  return (
                    <motion.div
                      key={entry.user_id}
                      className={`flex items-center justify-between rounded-md border px-3 py-2 ${
                        isLeader
                          ? "border-lime-300/60 bg-lime-300/10 shadow-[0_0_20px_rgba(163,230,53,0.18)]"
                          : "border-zinc-700/80 bg-zinc-950/65"
                      }`}
                      animate={isLeader ? { boxShadow: ["0 0 14px rgba(163,230,53,0.14)", "0 0 24px rgba(163,230,53,0.28)", "0 0 14px rgba(163,230,53,0.14)"] } : {}}
                      transition={isLeader ? { duration: 2.2, repeat: Infinity, ease: "easeInOut" } : {}}
                    >
                      <div className="inline-flex items-center gap-2">
                        <p className="inline-flex items-center gap-1.5 text-sm text-textMain">
                        {isLeader ? <Crown className="h-3.5 w-3.5 text-lime-300" /> : null}
                        #{entry.rank} {entry.username ?? entry.user_id.slice(0, 8)}
                        </p>
                        {isLeader ? (
                          <Badge variant="success" className="border-lime-300/75 bg-lime-300/20 text-lime-100">
                            Singularity
                          </Badge>
                        ) : null}
                      </div>
                      <p className={`text-sm font-semibold ${isLeader ? "text-lime-200 drop-shadow-[0_0_8px_rgba(163,230,53,0.52)]" : "text-lime-300"}`}>
                        {entry.rating.toFixed(1)} ELO
                      </p>
                    </motion.div>
                  );
                })}
                {leaderboardQuery.isLoading ? <p className="text-xs text-textDim">Actualizando top...</p> : null}
                {leaderboardQuery.isError ? <p className="text-xs text-redGlow">No se pudo cargar el top.</p> : null}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </AppShell>
  );
}
