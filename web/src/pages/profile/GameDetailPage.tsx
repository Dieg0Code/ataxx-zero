import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Bot, ChevronLeft, ChevronRight, Crown, Flag, Radar, Swords, User } from "lucide-react";
import { AppShell } from "@/widgets/layout/AppShell";
import { useAuth } from "@/app/providers/useAuth";
import { fetchPersistedReplay, type PersistedReplay } from "@/features/match/persistence";
import type { BoardState } from "@/features/match/types";
import { buildBoardTimeline, getOrderedReplayMoves } from "@/pages/profile/replayTimeline";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";

function renderMoveLabel(move: {
  r1: number | null;
  c1: number | null;
  r2: number | null;
  c2: number | null;
}): string {
  if (move.r1 === null || move.c1 === null || move.r2 === null || move.c2 === null) {
    return "pass";
  }
  return `(${move.r1},${move.c1}) -> (${move.r2},${move.c2})`;
}

function agentLabel(agent: string | undefined): string {
  if (!agent) {
    return "desconocido";
  }
  if (agent === "human") {
    return "humano";
  }
  if (agent === "model") {
    return "modelo";
  }
  if (agent === "random") {
    return "aleatorio";
  }
  if (agent === "heuristic") {
    return "heuristica";
  }
  return agent.split("_").join(" ");
}

function winnerLabel(winner: string | null): string {
  if (!winner || winner === "draw") {
    return "empate";
  }
  if (winner === "p1") {
    return "gana p1";
  }
  if (winner === "p2") {
    return "gana p2";
  }
  return winner;
}

function modeLabel(mode: string | null | undefined): string {
  if (!mode) {
    return "-";
  }
  if (mode === "manual") {
    return "manual";
  }
  if (mode === "fast") {
    return "rapido";
  }
  if (mode === "strong") {
    return "fuerte";
  }
  if (mode === "heuristic_easy") {
    return "heuristica easy";
  }
  if (mode === "heuristic_normal") {
    return "heuristica normal";
  }
  if (mode === "heuristic_hard") {
    return "heuristica hard";
  }
  return mode;
}

function boardAtPly(replay: PersistedReplay, selectedPly: number): BoardState | null {
  const timeline = buildBoardTimeline(replay);
  if (timeline.length === 0) {
    return null;
  }
  if (selectedPly <= 0) {
    return timeline[0];
  }
  const index = Math.min(selectedPly, timeline.length - 1);
  return timeline[index];
}

function cellClass(cell: number): string {
  if (cell === 1) {
    return "border-zinc-200 bg-zinc-100 shadow-[0_0_12px_rgba(255,255,255,0.35)]";
  }
  if (cell === -1) {
    return "border-lime-300 bg-lime-400 shadow-[0_0_14px_rgba(132,204,22,0.45)]";
  }
  return "border-zinc-700/80 bg-zinc-900/40";
}

export function GameDetailPage(): JSX.Element {
  const { gameId } = useParams<{ gameId: string }>();
  const { accessToken, loading, isAuthenticated, user } = useAuth();
  const [selectedPly, setSelectedPly] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timelineContainerRef = useRef<HTMLDivElement | null>(null);
  const timelineItemRefs = useRef<Record<number, HTMLButtonElement | null>>({});
  const userScrollLockRef = useRef(false);
  const userScrollUnlockTimerRef = useRef<number | null>(null);

  const replayQuery = useQuery({
    queryKey: ["profile-game-replay", gameId, accessToken],
    queryFn: () => fetchPersistedReplay(accessToken!, gameId!),
    enabled: Boolean(gameId && accessToken),
    retry: false,
    refetchOnWindowFocus: false,
  });

  const boardTimeline = useMemo(() => {
    if (!replayQuery.data) {
      return [];
    }
    return buildBoardTimeline(replayQuery.data);
  }, [replayQuery.data]);
  const orderedMoves = useMemo(() => {
    if (!replayQuery.data) {
      return [];
    }
    return getOrderedReplayMoves(replayQuery.data);
  }, [replayQuery.data]);
  const maxPly = Math.max(0, boardTimeline.length - 1);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    if (maxPly === 0 || selectedPly >= maxPly) {
      setIsPlaying(false);
      return;
    }
    const timeout = window.setTimeout(() => {
      setSelectedPly((prev) => Math.min(maxPly, prev + 1));
    }, 550);
    return () => window.clearTimeout(timeout);
  }, [isPlaying, maxPly, selectedPly]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
        return;
      }
      setIsPlaying(false);
      if (event.key === "ArrowLeft") {
        setSelectedPly((prev) => Math.max(0, prev - 1));
      } else {
        setSelectedPly((prev) => Math.min(maxPly, prev + 1));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [maxPly]);

  useEffect(() => {
    if (userScrollLockRef.current) {
      return;
    }
    if (selectedPly <= 0) {
      if (timelineContainerRef.current && typeof timelineContainerRef.current.scrollTo === "function") {
        timelineContainerRef.current.scrollTo({ top: 0, behavior: "smooth" });
      }
      return;
    }
    const container = timelineContainerRef.current;
    const target = timelineItemRefs.current[selectedPly];
    if (container && target && typeof container.scrollTo === "function") {
      const top =
        target.offsetTop - container.offsetTop - container.clientHeight / 2 + target.clientHeight / 2;
      container.scrollTo({
        top: Math.max(0, top),
        behavior: "smooth",
      });
    }
  }, [selectedPly]);

  useEffect(() => {
    return () => {
      if (userScrollUnlockTimerRef.current !== null) {
        window.clearTimeout(userScrollUnlockTimerRef.current);
        userScrollUnlockTimerRef.current = null;
      }
    };
  }, []);

  const boardState = useMemo(() => {
    if (!replayQuery.data) {
      return null;
    }
    return boardAtPly(replayQuery.data, selectedPly);
  }, [replayQuery.data, selectedPly]);

  const previousBoardState = useMemo(() => {
    if (!replayQuery.data) {
      return null;
    }
    return boardAtPly(replayQuery.data, Math.max(0, selectedPly - 1));
  }, [replayQuery.data, selectedPly]);

  const changedCells = useMemo(() => {
    const changed = new Set<string>();
    if (!boardState || !previousBoardState) {
      return changed;
    }
    for (let r = 0; r < boardState.grid.length; r += 1) {
      for (let c = 0; c < boardState.grid[r].length; c += 1) {
        if (boardState.grid[r][c] !== previousBoardState.grid[r][c]) {
          changed.add(`${r}:${c}`);
        }
      }
    }
    return changed;
  }, [boardState, previousBoardState]);

  const summary = useMemo(() => {
    if (!replayQuery.data) {
      return null;
    }
    const game = replayQuery.data.game;
    const isP1 = Boolean(user?.id && game.player1_id === user.id);
    const isP2 = Boolean(user?.id && game.player2_id === user.id);
    const mySide = isP1 ? "p1" : isP2 ? "p2" : null;
    const myAgent = isP1 ? game.player1_agent : isP2 ? game.player2_agent : null;
    const rivalSide = mySide === "p1" ? "p2" : mySide === "p2" ? "p1" : null;
    const rivalAgent = rivalSide === "p1" ? game.player1_agent : rivalSide === "p2" ? game.player2_agent : null;
    const myUsername =
      (isP1 ? game.player1_username : isP2 ? game.player2_username : null) ?? user?.username ?? "usuario";
    const rivalUsername = rivalSide === "p1" ? game.player1_username : rivalSide === "p2" ? game.player2_username : null;
    const p1Name = game.player1_username ?? (game.player1_id ? `p1:${game.player1_id.slice(0, 6)}` : "P1");
    const p2Name = game.player2_username ?? (game.player2_id ? `p2:${game.player2_id.slice(0, 6)}` : "P2");
    const winnerName =
      game.winner_side === "p1"
        ? p1Name
        : game.winner_side === "p2"
          ? p2Name
          : "Empate";
    const winnerToneClass =
      game.winner_side === "p2"
        ? "border-primary/55 bg-primary/12 text-primary"
        : game.winner_side === "p1"
          ? "border-zinc-300/50 bg-zinc-100/12 text-zinc-100"
          : "border-zinc-600/70 bg-zinc-900/65 text-zinc-200";
    const detectedModes = Array.from(
      new Set(orderedMoves.map((move) => move.mode).filter((mode) => mode !== "manual")),
    );
    const inferredRivalAgent =
      detectedModes.includes("random")
        ? "random"
        : detectedModes.some((mode) => mode.startsWith("heuristic_"))
          ? "heuristic"
          : detectedModes.some((mode) => mode === "fast" || mode === "strong")
            ? "model"
            : rivalAgent;
    const detectedAiMode =
      detectedModes.length > 0
        ? detectedModes.map((mode) => modeLabel(mode)).join(" / ")
        : inferredRivalAgent === "heuristic"
          ? "heuristica (no registrado)"
          : inferredRivalAgent === "model"
            ? "modelo (no registrado)"
            : "-";

    return {
      myUsername,
      rivalUsername,
      mySide,
      myAgent,
      rivalSide,
      rivalAgent: inferredRivalAgent,
      detectedAiMode,
      queueType: game.queue_type ?? "casual",
      rated: game.rated ?? false,
      p1Name,
      p2Name,
      winnerName,
      winnerToneClass,
      winnerText: winnerLabel(game.winner_side),
      termination: game.termination_reason ?? "normal",
    };
  }, [orderedMoves, replayQuery.data, user?.id, user?.username]);

  if (loading) {
    return (
      <AppShell>
        <p className="text-sm text-textDim">Cargando sesion...</p>
      </AppShell>
    );
  }
  if (!isAuthenticated) {
    return <Navigate to="/auth/login" replace />;
  }
  if (!gameId) {
    return <Navigate to="/profile" replace />;
  }

  return (
    <AppShell>
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-[0.15em] text-textDim">Replay</p>
            <h1 className="text-xl font-semibold text-textMain">Partida {gameId.slice(0, 8)}</h1>
          </div>
          <Button asChild variant="secondary" size="sm">
            <Link to="/profile">Volver a perfil</Link>
          </Button>
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Reproductor de tablero</CardTitle>
              <CardDescription>Navega la partida por ply para inspeccionar decisiones.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {replayQuery.isLoading ? <p className="text-textDim">Cargando replay...</p> : null}
              {replayQuery.isError ? (
                <p className="text-redGlow">No se pudo cargar el replay de esta partida.</p>
              ) : null}
              {replayQuery.data ? (
                <>
                  <div className="flex flex-wrap items-center gap-2 text-sm">
                    <Badge variant="default">status: {replayQuery.data.game.status}</Badge>
                    <Badge variant={replayQuery.data.game.winner_side ? "success" : "warning"}>
                      ganador: {summary?.winnerName ?? replayQuery.data.game.winner_side ?? "draw"}
                    </Badge>
                    <Badge variant="default">moves: {replayQuery.data.moves.length}</Badge>
                  </div>
                  {summary ? (
                    <div className="space-y-2">
                      <div className="rounded-md border border-line/70 bg-black/35 p-2.5">
                        <p className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-[0.14em] text-textDim">
                          <Crown className="h-3.5 w-3.5 text-primary" />
                          Resultado
                        </p>
                        <p className="mt-1.5 text-textMain">
                          Ganador:{" "}
                          <span
                            className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-semibold ${summary.winnerToneClass}`}
                          >
                            {summary.winnerName}
                          </span>
                        </p>
                        <p className="mt-1 text-[12px] text-textDim">
                          {summary.winnerText} · fin por {summary.termination}
                        </p>
                      </div>
                    <div className="grid gap-2 rounded-lg border border-line/70 bg-bg1/60 p-3 text-sm sm:grid-cols-2">
                      <p className="text-textDim">
                        <span className="inline-flex items-center gap-1.5">
                          <User className="h-3.5 w-3.5 text-zinc-200" />
                          Jugador:
                        </span>{" "}
                        <span className="font-medium text-textMain">
                          {summary.myUsername} {summary.mySide ? `(${summary.mySide})` : ""}
                        </span>
                      </p>
                      <p className="text-textDim">
                        <span className="inline-flex items-center gap-1.5">
                          <Bot className="h-3.5 w-3.5 text-primary" />
                          Rival:
                        </span>{" "}
                        <span className="font-medium text-textMain">
                          {summary.rivalUsername ? `${summary.rivalUsername} · ` : ""}
                          {summary.rivalAgent ? `${agentLabel(summary.rivalAgent)}${summary.rivalSide ? ` (${summary.rivalSide})` : ""}` : "desconocido"}
                        </span>
                      </p>
                      <p className="text-textDim">
                        <span className="inline-flex items-center gap-1.5">
                          <Swords className="h-3.5 w-3.5 text-primary/90" />
                          Agentes:
                        </span>{" "}
                        <span className="font-medium text-textMain">
                          {agentLabel(replayQuery.data.game.player1_agent)} vs {agentLabel(replayQuery.data.game.player2_agent)}
                        </span>
                      </p>
                      <p className="text-textDim">
                        <span className="inline-flex items-center gap-1.5">
                          <Radar className="h-3.5 w-3.5 text-primary" />
                          Modo IA detectado:
                        </span>{" "}
                        <span className="font-medium text-primary">{summary.detectedAiMode}</span>
                      </p>
                      <p className="text-textDim">
                        <span className="inline-flex items-center gap-1.5">
                          <Flag className="h-3.5 w-3.5 text-zinc-200" />
                          Cola:
                        </span>{" "}
                        <span className="font-medium text-textMain">
                          {summary.queueType} {summary.rated ? "(ranked)" : "(casual)"}
                        </span>
                      </p>
                      <p className="text-textDim">
                        Resultado:{" "}
                        <span className="font-medium text-textMain">
                          {summary.winnerText} · fin por {summary.termination}
                        </span>
                      </p>
                    </div>
                    </div>
                  ) : null}

                  <div className="rounded-lg border border-line/70 bg-black/35 p-3">
                    <div className="mb-3 flex items-center justify-between text-sm">
                      <p className="text-textMain">
                        {selectedPly === 0 ? "Estado inicial" : `Despues de ply ${selectedPly - 1}`}
                      </p>
                      <p className="text-textDim">Paso {selectedPly}/{maxPly}</p>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={maxPly}
                      value={selectedPly}
                      onChange={(e) => setSelectedPly(Number(e.target.value))}
                      className="w-full accent-lime-400"
                      disabled={maxPly === 0}
                    />
                    <div className="mt-3 flex items-center gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        disabled={maxPly === 0}
                        onClick={() => setIsPlaying((prev) => !prev)}
                      >
                        {isPlaying ? "Pausar" : "Reproducir"}
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        aria-label="Paso anterior"
                        disabled={selectedPly <= 0}
                        onClick={() => {
                          setIsPlaying(false);
                          setSelectedPly((prev) => Math.max(0, prev - 1));
                        }}
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        aria-label="Paso siguiente"
                        disabled={selectedPly >= maxPly}
                        onClick={() => {
                          setIsPlaying(false);
                          setSelectedPly((prev) => Math.min(maxPly, prev + 1));
                        }}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          setIsPlaying(false);
                          setSelectedPly(0);
                        }}
                      >
                        Inicio
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          setIsPlaying(false);
                          setSelectedPly(maxPly);
                        }}
                      >
                        Final
                      </Button>
                    </div>
                  </div>

                  {boardState ? (
                    <motion.div
                      className="grid grid-cols-7 gap-1 rounded-lg border border-line/70 bg-black/35 p-2"
                      key={`board-${selectedPly}`}
                      initial={{ opacity: 0.94 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.14 }}
                    >
                      {boardState.grid.map((row, rowIdx) =>
                        row.map((cell, colIdx) => (
                          <motion.div
                            key={`${rowIdx}:${colIdx}`}
                            className="flex aspect-square items-center justify-center rounded border border-zinc-800 bg-zinc-950/70"
                            animate={
                              changedCells.has(`${rowIdx}:${colIdx}`)
                                ? {
                                    borderColor: ["rgba(63,63,70,1)", "rgba(163,230,53,0.95)", "rgba(63,63,70,1)"],
                                    boxShadow: [
                                      "0 0 0px rgba(132,204,22,0)",
                                      "0 0 18px rgba(132,204,22,0.55)",
                                      "0 0 0px rgba(132,204,22,0)",
                                    ],
                                  }
                                : {}
                            }
                            transition={{ duration: 0.45, ease: "easeOut" }}
                          >
                            {cell !== 0 ? (
                              <motion.span
                                key={`${rowIdx}:${colIdx}:${selectedPly}:${cell}`}
                                className={`h-4/5 w-4/5 rounded-full border-2 ${cellClass(cell)}`}
                                initial={{ scale: changedCells.has(`${rowIdx}:${colIdx}`) ? 0.72 : 1, opacity: 0.9 }}
                                animate={{ scale: 1, opacity: 1 }}
                                transition={{ type: "spring", stiffness: 280, damping: 22 }}
                              />
                            ) : null}
                          </motion.div>
                        )),
                      )}
                    </motion.div>
                  ) : (
                    <p className="text-sm text-textDim">Sin estado de tablero disponible para esta partida.</p>
                  )}
                </>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Timeline de jugadas</CardTitle>
              <CardDescription>Selecciona una jugada para saltar directo a ese punto.</CardDescription>
            </CardHeader>
            <CardContent>
              {orderedMoves.length === 0 ? (
                <p className="text-sm text-textDim">Aun no hay jugadas registradas.</p>
              ) : (
                <div
                  ref={timelineContainerRef}
                  className="replay-scroll max-h-[640px] space-y-2 overflow-auto pr-1"
                  onScroll={() => {
                    userScrollLockRef.current = true;
                    if (userScrollUnlockTimerRef.current !== null) {
                      window.clearTimeout(userScrollUnlockTimerRef.current);
                    }
                    userScrollUnlockTimerRef.current = window.setTimeout(() => {
                      userScrollLockRef.current = false;
                      userScrollUnlockTimerRef.current = null;
                    }, 900);
                  }}
                >
                  {orderedMoves.map((move, idx) => {
                    const selected = idx + 1 === selectedPly;
                    const sideLabel = move.player_side === "p1" ? "P1" : "P2";
                    return (
                      <motion.button
                        key={`${move.ply}-${move.player_side}-${idx}`}
                        ref={(node) => { timelineItemRefs.current[idx + 1] = node; }}
                        type="button"
                        onClick={() => setSelectedPly(idx + 1)}
                        aria-current={selected ? "step" : undefined}
                        className={`group relative w-full overflow-hidden rounded-lg border px-3 py-2.5 text-left text-sm transition ${
                          selected
                            ? "border-lime-300/90 bg-gradient-to-r from-lime-300/28 via-lime-300/12 to-transparent ring-2 ring-lime-300/50 shadow-[0_0_24px_rgba(163,230,53,0.30)]"
                            : "border-zinc-700/85 bg-black/45 hover:border-lime-300/50 hover:bg-bg1/90"
                        }`}
                        animate={
                          selected
                            ? {
                                boxShadow: [
                                  "0 0 0px rgba(156,226,51,0)",
                                  "0 0 18px rgba(156,226,51,0.26)",
                                  "0 0 0px rgba(156,226,51,0)",
                                ],
                              }
                            : {}
                        }
                        transition={{ duration: 1.2, repeat: selected ? Infinity : 0, ease: "easeInOut" }}
                      >
                        {selected ? (
                          <motion.span
                            className="pointer-events-none absolute inset-y-1 left-1 w-1 rounded-full bg-primary/90"
                            initial={{ opacity: 0.4 }}
                            animate={{ opacity: [0.45, 1, 0.45] }}
                            transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                          />
                        ) : null}
                        <div className="flex items-center justify-between gap-2">
                          <p className="font-medium text-textMain">Ply {move.ply}</p>
                          <div className="flex items-center gap-1.5">
                            {selected ? (
                              <span className="rounded-full border border-lime-300/80 bg-lime-300/25 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-lime-200">
                                En curso
                              </span>
                            ) : null}
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] ${
                                move.player_side === "p1"
                                  ? "border-zinc-300/60 bg-zinc-100/10 text-zinc-100"
                                  : "border-primary/60 bg-primary/10 text-primary"
                              }`}
                            >
                              {sideLabel}
                            </span>
                          </div>
                        </div>
                        <p className="mt-1 font-mono text-[13px] text-textMain/95">{renderMoveLabel(move)}</p>
                        <div className="mt-1.5 flex items-center justify-between text-[11px] uppercase tracking-[0.12em]">
                          <span className="text-textDim">modo</span>
                          <span className="text-primary/90">{modeLabel(move.mode)}</span>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}







