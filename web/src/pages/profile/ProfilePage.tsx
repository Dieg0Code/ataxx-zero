import { useEffect, useMemo, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowUpRight,
  CalendarDays,
  CircleCheck,
  CircleDot,
  CircleX,
  Eye,
  Hash,
  History,
  Hourglass,
  Mail,
  PauseCircle,
  Shield,
  Sparkles,
  Swords,
  Trash2,
  TrendingDown,
  TrendingUp,
  User as UserIcon,
} from "lucide-react";
import { AppShell } from "@/widgets/layout/AppShell";
import { useAuth } from "@/app/providers/useAuth";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { deleteMyGame, fetchMyGames, type ProfileGame } from "@/features/profile/api";
import { InvitationList } from "@/features/matches/InvitationList";
import { useInvitations } from "@/features/matches/useInvitations";
import {
  fetchActiveSeason,
  fetchRatingEvents,
  fetchUserRating,
  type RatingEventResponse,
} from "@/features/ranking/api";

const PAGE_SIZE = 8;
const MATCHMAKING_MATCH_KEY = "ataxx.matchmaking.match.v1";

function isSpectatorView(game: ProfileGame, userId: string): boolean {
  const isParticipant = game.player1_id === userId || game.player2_id === userId;
  const isAiVsAi = game.player1_agent !== "human" && game.player2_agent !== "human";
  return isAiVsAi && !isParticipant;
}

function outcomeLabel(game: ProfileGame, userId: string): string {
  const spectatorView = isSpectatorView(game, userId);
  if (game.status !== "finished") {
    return spectatorView ? "Partida IA" : "En curso";
  }
  if (game.winner_user_id === null) {
    return spectatorView ? "Empate IA" : "Empate";
  }
  if (spectatorView) {
    return "Resultado IA";
  }
  return game.winner_user_id === userId ? "Victoria" : "Derrota";
}

function outcomeVisual(game: ProfileGame, userId: string): { label: string; icon: JSX.Element } {
  const label = outcomeLabel(game, userId);
  if (label === "Victoria") {
    return { label, icon: <CircleCheck className="h-3.5 w-3.5" /> };
  }
  if (label === "Derrota") {
    return { label, icon: <CircleX className="h-3.5 w-3.5" /> };
  }
  if (label === "Empate" || label === "Empate IA") {
    return { label, icon: <PauseCircle className="h-3.5 w-3.5" /> };
  }
  if (label === "Resultado IA" || label === "Partida IA") {
    return { label, icon: <Eye className="h-3.5 w-3.5" /> };
  }
  return { label, icon: <Hourglass className="h-3.5 w-3.5" /> };
}

function outcomeTone(label: string): {
  marker: string;
  outcomeText: string;
  outcomeGlow: string;
  statusText: string;
} {
  if (label === "Victoria") {
    return {
      marker: "bg-lime-400/90 shadow-[0_0_14px_rgba(163,230,53,0.55)]",
      outcomeText: "text-lime-200",
      outcomeGlow: "text-lime-300 drop-shadow-[0_0_11px_rgba(163,230,53,0.52)]",
      statusText: "text-lime-300/80",
    };
  }
  if (label === "Derrota") {
    return {
      marker: "bg-red-500/90 shadow-[0_0_14px_rgba(239,68,68,0.55)]",
      outcomeText: "text-red-300",
      outcomeGlow: "text-red-400 drop-shadow-[0_0_11px_rgba(239,68,68,0.52)]",
      statusText: "text-red-300/80",
    };
  }
  if (label === "Empate" || label === "Empate IA") {
    return {
      marker: "bg-zinc-400/55",
      outcomeText: "text-zinc-200",
      outcomeGlow: "text-zinc-300",
      statusText: "text-zinc-400/80",
    };
  }
  if (label === "Resultado IA" || label === "Partida IA") {
    return {
      marker: "bg-cyan-300/65",
      outcomeText: "text-cyan-200",
      outcomeGlow: "text-cyan-300/85",
      statusText: "text-cyan-300/65",
    };
  }
  return {
    marker: "bg-amber-300/65",
    outcomeText: "text-amber-200",
    outcomeGlow: "text-amber-300/85",
    statusText: "text-amber-300/65",
  };
}

function statusVisual(status: string): { label: string; icon: JSX.Element } {
  const label = statusLabel(status);
  if (status === "finished") {
    return { label, icon: <CircleCheck className="h-3.5 w-3.5" /> };
  }
  if (status === "aborted") {
    return { label, icon: <CircleX className="h-3.5 w-3.5" /> };
  }
  if (status === "in_progress") {
    return { label, icon: <CircleDot className="h-3.5 w-3.5" /> };
  }
  return { label, icon: <Hourglass className="h-3.5 w-3.5" /> };
}

function queueLabel(queueType: string): string {
  return queueType === "ranked" ? "Ranked" : "Casual";
}

function statusLabel(status: string): string {
  switch (status) {
    case "pending":
      return "Pendiente";
    case "in_progress":
      return "En curso";
    case "finished":
      return "Finalizada";
    case "aborted":
      return "Abortada";
    default:
      return status;
  }
}

function transitionLabel(event: RatingEventResponse): string {
  if (event.transition_type === "promotion") {
    return event.major_promo_name ? `Promo: ${event.major_promo_name}` : "Ascenso";
  }
  if (event.transition_type === "demotion") {
    return "Descenso";
  }
  return "";
}

export function ProfilePage(): JSX.Element {
  const { user, loading, accessToken } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [offset, setOffset] = useState(0);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [pendingDeleteGame, setPendingDeleteGame] = useState<ProfileGame | null>(null);
  const queryClient = useQueryClient();
  const emitFlash = (message: string, tone: "success" | "warning" | "error" | "info"): void => {
    navigate(`${location.pathname}${location.search}${location.hash}`, {
      replace: true,
      state: { flash: { message, tone } },
    });
  };

  const createdAt = useMemo(() => {
    if (!user?.created_at) {
      return "-";
    }
    return new Date(user.created_at).toLocaleString();
  }, [user?.created_at]);

  const gamesQuery = useQuery({
    queryKey: ["profile-games", offset, accessToken],
    queryFn: () => fetchMyGames(accessToken!, PAGE_SIZE, offset, ["finished"]),
    enabled: Boolean(accessToken),
  });
  const {
    invitations: invitationItems,
    isLoading: invitationsLoading,
    isError: invitationsError,
    actionLoadingId: invitationActionLoadingId,
    acceptInvitationById,
    rejectInvitationById,
  } = useInvitations({
    accessToken,
    enabled: Boolean(accessToken),
    includeInitialFetch: true,
    scope: "profile",
  });

  const activeSeasonQuery = useQuery({
    queryKey: ["active-season", accessToken],
    queryFn: fetchActiveSeason,
  });

  const ratingQuery = useQuery({
    queryKey: ["profile-rating", user?.id, activeSeasonQuery.data?.id],
    queryFn: () => fetchUserRating(user!.id, activeSeasonQuery.data!.id),
    enabled: Boolean(user?.id && activeSeasonQuery.data?.id),
  });

  const eventsQuery = useQuery({
    queryKey: ["profile-rating-events", user?.id, activeSeasonQuery.data?.id],
    queryFn: () => fetchRatingEvents(user!.id, activeSeasonQuery.data!.id, 6, 0),
    enabled: Boolean(user?.id && activeSeasonQuery.data?.id),
  });

  useEffect(() => {
    const body = document.body;
    const previousOverflow = body.style.overflow;
    if (pendingDeleteGame !== null) {
      body.style.overflow = "hidden";
    }
    return () => {
      body.style.overflow = previousOverflow || "";
    };
  }, [pendingDeleteGame]);
  const deleteGameMutation = useMutation({
    mutationFn: async (gameId: string) => {
      if (!accessToken) {
        throw new Error("Sesion no valida.");
      }
      await deleteMyGame(accessToken, gameId);
    },
    onSuccess: async () => {
      setDeleteError(null);
      setPendingDeleteGame(null);
      await queryClient.invalidateQueries({ queryKey: ["profile-games"] });
      emitFlash("Replay eliminada correctamente.", "success");
    },
    onError: (error: unknown) => {
      const message = error instanceof Error ? error.message : "No se pudo eliminar la replay.";
      setDeleteError(message);
      emitFlash(message, "error");
    },
  });
  const acceptInvitationMutation = useMutation({
    mutationFn: async (gameId: string) => {
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
    },
    onSuccess: () => {
      emitFlash("Invitacion aceptada. Entrando a la arena...", "success");
      navigate("/match?queue=1");
    },
    onError: (error: unknown) => {
      const message = error instanceof Error ? error.message : "No se pudo aceptar la invitacion.";
      emitFlash(message, "error");
    },
  });
  const rejectInvitationMutation = useMutation({
    mutationFn: async (gameId: string) => {
      await rejectInvitationById(gameId);
    },
    onSuccess: () => {
      emitFlash("Invitacion rechazada.", "info");
    },
    onError: (error: unknown) => {
      const message = error instanceof Error ? error.message : "No se pudo rechazar la invitacion.";
      emitFlash(message, "error");
    },
  });

  useEffect(() => {
    if (pendingDeleteGame === null) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape" && !deleteGameMutation.isPending) {
        setPendingDeleteGame(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteGameMutation.isPending, pendingDeleteGame]);

  if (loading) {
    return (
      <AppShell>
        <p className="text-sm text-textDim">Cargando perfil...</p>
      </AppShell>
    );
  }
  if (user === null) {
    return <Navigate to="/auth/login" replace />;
  }

  return (
    <AppShell>
      <AnimatePresence>
        {pendingDeleteGame ? (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
            onClick={() => {
              if (!deleteGameMutation.isPending) {
                setPendingDeleteGame(null);
              }
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            <motion.div
              className="w-full max-w-md rounded-xl border border-zinc-700/80 bg-zinc-950/95 p-4 shadow-[0_0_32px_rgba(0,0,0,0.55)]"
              onClick={(event) => event.stopPropagation()}
              initial={{ opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <p className="text-xs uppercase tracking-[0.14em] text-textDim">Eliminar replay</p>
              <h3 className="mt-1 text-lg font-semibold text-textMain">Confirmar eliminacion</h3>
              <p className="mt-2 text-sm text-textDim">
                Esta accion eliminara la replay <span className="font-medium text-textMain">{pendingDeleteGame.id.slice(0, 8)}</span> y no se puede deshacer.
              </p>
              <div className="mt-4 flex items-center justify-end gap-2">
                <Button type="button" variant="secondary" size="sm" onClick={() => setPendingDeleteGame(null)} disabled={deleteGameMutation.isPending}>
                  Cancelar
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="border border-red-500/55 bg-red-500/15 text-red-200 hover:bg-red-500/25"
                  onClick={() => deleteGameMutation.mutate(pendingDeleteGame.id)}
                  disabled={deleteGameMutation.isPending}
                >
                  {deleteGameMutation.isPending ? "Eliminando..." : "Si, eliminar"}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card className="border-zinc-800/90 bg-zinc-950/60">
          <CardHeader>
            <CardTitle>Mi perfil</CardTitle>
            <CardDescription>Identidad de jugador y estado de cuenta.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="relative overflow-hidden rounded-md border border-lime-400/30 bg-zinc-950/75 px-3 py-2 shadow-[0_0_16px_rgba(163,230,53,0.06)] transition-all duration-300 hover:border-lime-300/55 hover:shadow-[0_0_24px_rgba(163,230,53,0.14)]">
                <motion.span
                  className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lime-300/70 to-transparent"
                  initial={{ x: "-35%", opacity: 0.38 }}
                  animate={{ x: ["-35%", "35%", "-35%"], opacity: [0.24, 0.5, 0.24] }}
                  transition={{ duration: 4.4, ease: "easeInOut", repeat: Infinity }}
                />
                <p className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-textDim">
                  <UserIcon className="h-3.5 w-3.5 text-lime-300" />Usuario
                </p>
                <p className="mt-1 text-sm font-medium text-textMain">{user.username}</p>
              </div>
              <div className="relative overflow-hidden rounded-md border border-lime-400/30 bg-zinc-950/75 px-3 py-2 shadow-[0_0_16px_rgba(163,230,53,0.06)] transition-all duration-300 hover:border-lime-300/55 hover:shadow-[0_0_24px_rgba(163,230,53,0.14)]">
                <motion.span
                  className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lime-300/70 to-transparent"
                  initial={{ x: "35%", opacity: 0.38 }}
                  animate={{ x: ["35%", "-35%", "35%"], opacity: [0.24, 0.5, 0.24] }}
                  transition={{ duration: 4.8, ease: "easeInOut", repeat: Infinity }}
                />
                <p className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-textDim">
                  <Mail className="h-3.5 w-3.5 text-lime-300" />Email
                </p>
                <p className="mt-1 text-sm font-medium text-textMain">{user.email ?? "-"}</p>
              </div>
              <div className="relative overflow-hidden rounded-md border border-lime-400/30 bg-zinc-950/75 px-3 py-2 shadow-[0_0_16px_rgba(163,230,53,0.06)] transition-all duration-300 hover:border-lime-300/55 hover:shadow-[0_0_24px_rgba(163,230,53,0.14)]">
                <motion.span
                  className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lime-300/70 to-transparent"
                  initial={{ x: "-25%", opacity: 0.4 }}
                  animate={{ x: ["-25%", "25%", "-25%"], opacity: [0.24, 0.52, 0.24] }}
                  transition={{ duration: 5, ease: "easeInOut", repeat: Infinity }}
                />
                <p className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-textDim">
                  <CalendarDays className="h-3.5 w-3.5 text-lime-300" />Alta
                </p>
                <p className="mt-1 text-sm font-medium text-textMain">{createdAt}</p>
              </div>
              <div className="relative overflow-hidden rounded-md border border-lime-400/30 bg-zinc-950/75 px-3 py-2 shadow-[0_0_16px_rgba(163,230,53,0.06)] transition-all duration-300 hover:border-lime-300/55 hover:shadow-[0_0_24px_rgba(163,230,53,0.14)]">
                <motion.span
                  className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lime-300/70 to-transparent"
                  initial={{ x: "20%", opacity: 0.36 }}
                  animate={{ x: ["20%", "-20%", "20%"], opacity: [0.22, 0.46, 0.22] }}
                  transition={{ duration: 5.2, ease: "easeInOut", repeat: Infinity }}
                />
                <p className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-textDim">
                  <Hash className="h-3.5 w-3.5 text-lime-300" />ID jugador
                </p>
                <p className="mt-1 truncate font-mono text-xs text-textMain">{user.id}</p>
              </div>
            </div>
            <div className="flex gap-2 pt-1">
              <Badge variant={user.is_active ? "success" : "warning"}>{user.is_active ? "Activo" : "Inactivo"}</Badge>
              {user.is_admin ? (
                <Badge variant="default" className="gap-1">
                  <Shield className="h-3 w-3" />Admin
                </Badge>
              ) : null}
            </div>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden border-zinc-800/90 bg-zinc-950/60">
          <motion.div
            aria-hidden="true"
            className="pointer-events-none absolute -right-10 top-5 h-24 w-24 rounded-full bg-lime-300/12 blur-2xl"
            initial={{ opacity: 0.28, scale: 0.92 }}
            animate={{ opacity: [0.22, 0.44, 0.22], scale: [0.92, 1.05, 0.92] }}
            transition={{ duration: 4.2, repeat: Infinity, ease: "easeInOut" }}
          />
          <CardHeader>
            <CardTitle className="text-base">Estado competitivo</CardTitle>
            <CardDescription>Lectura rapida de tu progreso en temporada activa.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {activeSeasonQuery.isLoading || ratingQuery.isLoading ? <p className="text-sm text-textDim">Sincronizando ladder...</p> : null}
            {activeSeasonQuery.isError || ratingQuery.isError ? <p className="text-sm text-redGlow">No se pudo cargar el estado competitivo.</p> : null}

            {ratingQuery.data ? (
              <>
                <motion.div
                  className="relative overflow-hidden rounded-md border border-lime-400/50 bg-lime-400/10 p-3 shadow-[0_0_24px_rgba(163,230,53,0.12)]"
                  initial={{ opacity: 0.94, scale: 1 }}
                  animate={{
                    opacity: [0.9, 1, 0.9],
                    scale: [1, 1.007, 1],
                    boxShadow: [
                      "0 0 20px rgba(163,230,53,0.12)",
                      "0 0 30px rgba(163,230,53,0.2)",
                      "0 0 20px rgba(163,230,53,0.12)",
                    ],
                  }}
                  transition={{
                    duration: 2.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    times: [0, 0.42, 1],
                  }}
                >
                  <motion.span
                    className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lime-300/75 to-transparent"
                    initial={{ x: "-40%", opacity: 0.4 }}
                    animate={{ x: ["-40%", "40%", "-40%"], opacity: [0.24, 0.6, 0.24] }}
                    transition={{ duration: 3.1, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <p className="text-[11px] uppercase tracking-[0.14em] text-textDim">Liga actual</p>
                  <div className="mt-1 flex items-center justify-between">
                    <p className="flex items-center gap-2 text-base font-semibold text-lime-200">
                      <Sparkles className="h-4 w-4 text-lime-300 drop-shadow-[0_0_8px_rgba(163,230,53,0.9)]" />
                      {ratingQuery.data.league} {ratingQuery.data.division}
                    </p>
                    <motion.div
                      initial={{ scale: 0.94, opacity: 0.85 }}
                      animate={{ scale: [0.94, 1.06, 1], opacity: 1 }}
                      transition={{ duration: 0.36, ease: "easeOut" }}
                    >
                      <Badge
                        variant="success"
                        className="border-lime-300/70 bg-lime-300/25 text-lime-100 shadow-[0_0_18px_rgba(163,230,53,0.28)]"
                      >
                        {ratingQuery.data.lp} LP
                      </Badge>
                    </motion.div>
                  </div>
                  {ratingQuery.data.next_major_promo ? (
                    <p className="mt-2 flex items-center gap-2 text-xs text-amber">
                      <ArrowUpRight className="h-3.5 w-3.5" />Siguiente major promo: {ratingQuery.data.next_major_promo}
                    </p>
                  ) : null}
                </motion.div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 px-2 py-2 text-center">
                    <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">MMR</p>
                    <p className="mt-1 text-sm font-semibold text-lime-200 drop-shadow-[0_0_8px_rgba(163,230,53,0.45)]">
                      {ratingQuery.data.rating.toFixed(1)}
                    </p>
                  </div>
                  <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 px-2 py-2 text-center">
                    <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">W/L/D</p>
                    <p className="mt-1 text-sm font-semibold text-textMain">{ratingQuery.data.wins}/{ratingQuery.data.losses}/{ratingQuery.data.draws}</p>
                  </div>
                  <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 px-2 py-2 text-center">
                    <p className="text-[11px] uppercase tracking-[0.12em] text-textDim">Partidas</p>
                    <p className="mt-1 text-sm font-semibold text-textMain">{ratingQuery.data.games_played}</p>
                  </div>
                </div>
              </>
            ) : null}
          </CardContent>
        </Card>
        <Card className="border-zinc-800/90 bg-zinc-950/60">
          <CardHeader>
            <CardTitle className="text-base">Invitaciones 1v1</CardTitle>
            <CardDescription>Duelos directos pendientes para tu cuenta.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <InvitationList
              items={invitationItems}
              isLoading={invitationsLoading}
              isError={invitationsError}
              actionLoadingId={invitationActionLoadingId}
              variant="card"
              onAccept={(gameId) => acceptInvitationMutation.mutate(gameId)}
              onReject={(gameId) => rejectInvitationMutation.mutate(gameId)}
            />
          </CardContent>
        </Card>

        <Card className="border-zinc-800/90 bg-zinc-950/60 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-4.5 w-4.5 text-lime-300" />Actividad de ladder
            </CardTitle>
            <CardDescription>Ultimos cambios de rating en temporada activa.</CardDescription>
          </CardHeader>
          <CardContent>
            {eventsQuery.isLoading ? <p className="text-sm text-textDim">Cargando actividad...</p> : null}
            {eventsQuery.isError ? <p className="text-sm text-redGlow">No se pudo cargar la actividad de ladder.</p> : null}
            {!eventsQuery.isLoading && (eventsQuery.data?.items.length ?? 0) === 0 ? (
              <p className="text-sm text-textDim">Aun no hay eventos competitivos para mostrar.</p>
            ) : null}

            {(eventsQuery.data?.items ?? []).length > 0 ? (
              <div className="profile-scroll max-h-72 space-y-2 overflow-y-auto pr-1">
                {(eventsQuery.data?.items ?? []).map((event) => (
                  <div key={event.id} className="flex items-center justify-between rounded-md border border-zinc-700/80 bg-zinc-950/75 px-3 py-2">
                    {(() => {
                      const lpDelta =
                        event.before_lp !== null && event.after_lp !== null
                          ? event.after_lp - event.before_lp
                          : null;
                      const primaryDelta = lpDelta ?? event.delta;
                      const primaryIsLp = lpDelta !== null;
                      return (
                        <>
                    <div className="min-w-0">
                      <p className="flex items-center gap-2 text-sm text-textMain">
                        {primaryDelta >= 0 ? <TrendingUp className="h-3.5 w-3.5 text-lime-300" /> : <TrendingDown className="h-3.5 w-3.5 text-redGlow" />}
                        {event.before_league} {event.before_division} -&gt; {event.after_league} {event.after_division}
                      </p>
                      <p className="mt-1 text-xs text-textDim">{new Date(event.created_at).toLocaleString()} - game {event.game_id.slice(0, 8)}</p>
                    </div>
                    <div className="ml-3 flex items-center gap-2">
                      {transitionLabel(event) ? <span className="text-xs text-textDim">{transitionLabel(event)}</span> : null}
                      <div className="text-right">
                        <p className="text-[10px] uppercase tracking-[0.12em] text-textDim">
                          {primaryIsLp ? "LP de liga" : "Delta MMR"}
                        </p>
                        <motion.span
                          className={`text-sm font-semibold ${
                            primaryDelta >= 0
                            ? "text-lime-300 drop-shadow-[0_0_8px_rgba(163,230,53,0.45)]"
                            : "text-red-300"
                          }`}
                          initial={{ opacity: 0.8, y: 0 }}
                          animate={primaryDelta >= 0 ? { opacity: [0.82, 1, 0.82], y: [0, -0.5, 0] } : {}}
                          transition={{ duration: 1.8, repeat: primaryDelta >= 0 ? Infinity : 0, ease: "easeInOut" }}
                        >
                          {primaryDelta >= 0 ? "+" : ""}
                          {primaryIsLp ? primaryDelta.toFixed(0) : primaryDelta.toFixed(1)} {primaryIsLp ? "LP" : "MMR"}
                        </motion.span>
                        {primaryIsLp ? (
                          <p className="mt-0.5 text-[11px] text-textDim">
                            MMR paralelo {event.delta >= 0 ? "+" : ""}
                            {event.delta.toFixed(1)}
                          </p>
                        ) : null}
                      </div>
                    </div>
                        </>
                      );
                    })()}
                  </div>
                ))}
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card className="border-zinc-800/90 bg-zinc-950/60 lg:col-span-2">
          <CardHeader>
            <CardTitle>Historial de partidas</CardTitle>
            <CardDescription>Solo partidas finalizadas de tu cuenta autenticada.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {gamesQuery.isLoading ? <p className="text-sm text-textDim">Cargando partidas...</p> : null}
            {gamesQuery.isError ? <p className="text-sm text-redGlow">No se pudo cargar el historial de partidas.</p> : null}
            {deleteError ? <p className="text-sm text-redGlow">{deleteError}</p> : null}
            {!gamesQuery.isLoading && !gamesQuery.isError && (gamesQuery.data?.items.length ?? 0) === 0 ? (
              <p className="text-sm text-textDim">Todavia no tienes partidas registradas.</p>
            ) : null}

            {(gamesQuery.data?.items ?? []).map((game) => {
              const outcome = outcomeVisual(game, user.id);
              const status = statusVisual(game.status);
              const tone = outcomeTone(outcome.label);
              return (
                <div
                  key={game.id}
                  className="relative grid gap-2 rounded-md border border-zinc-700/80 bg-zinc-950/75 px-3 py-2 text-sm md:grid-cols-[1fr_auto_auto]"
                >
                  <span aria-hidden="true" className={`absolute bottom-2 left-1 top-2 w-[2px] rounded-full ${tone.marker}`} />
                  <div className="min-w-[180px]">
                    <p className="flex items-center gap-2 text-textMain">
                      <Swords className="h-3.5 w-3.5 text-lime-300" />{game.id.slice(0, 8)}
                    </p>
                    <p className="mt-1 text-xs text-textDim">
                      {queueLabel(game.queue_type)} - {isSpectatorView(game, user.id) ? "espectador" : game.rated ? "con ELO" : "sin ELO"}
                    </p>
                  </div>

                  <div className="flex min-w-[180px] items-center gap-3">
                    <span
                      aria-label={`Resultado: ${outcome.label}`}
                      title={outcome.label}
                      className={`inline-flex h-7 w-7 items-center justify-center ${tone.outcomeGlow}`}
                    >
                      {outcome.icon}
                    </span>
                    <div className="leading-tight">
                      <p className={`text-[11px] font-semibold uppercase tracking-[0.14em] ${tone.outcomeText}`}>{outcome.label}</p>
                      <p className="mt-0.5 flex items-center gap-1 text-[11px] uppercase tracking-[0.12em] text-textDim">
                        <span
                          aria-label={`Estado: ${status.label}`}
                          title={status.label}
                          className={`inline-flex h-4 w-4 items-center justify-center ${tone.statusText}`}
                        >
                          {status.icon}
                        </span>
                        {status.label}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center justify-end gap-4">
                    <Button
                      asChild
                      type="button"
                      size="sm"
                      variant="ghost"
                      className="h-8 px-0 text-lime-200 transition hover:bg-transparent hover:text-lime-100"
                    >
                      <Link to={`/profile/games/${game.id}`} aria-label={`Detalle ${game.id.slice(0, 8)}`}>
                        <motion.span
                          className="inline-flex items-center gap-1 text-xs tracking-[0.12em] uppercase"
                          initial={{ opacity: 0.86 }}
                          animate={{ opacity: [0.86, 1, 0.86] }}
                          transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                        >
                          Detalle
                          <motion.span
                            initial={{ x: 0 }}
                            animate={{ x: [0, 2, 0] }}
                            transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                          >
                            <ArrowUpRight className="h-3.5 w-3.5 text-lime-300" />
                          </motion.span>
                        </motion.span>
                      </Link>
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      aria-label={`Eliminar ${game.id.slice(0, 8)}`}
                      className="h-8 w-8 p-0 text-red-300/85 transition hover:bg-transparent hover:text-red-200"
                      disabled={deleteGameMutation.isPending}
                      onClick={() => setPendingDeleteGame(game)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              );
            })}

            <div className="flex items-center justify-between pt-1">
              <Button type="button" variant="secondary" size="sm" disabled={offset === 0} onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}>
                Anterior
              </Button>
              <p className="text-xs text-textDim">offset: {offset}</p>
              <Button type="button" variant="secondary" size="sm" disabled={!gamesQuery.data?.has_more} onClick={() => setOffset((prev) => prev + PAGE_SIZE)}>
                Siguiente
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
