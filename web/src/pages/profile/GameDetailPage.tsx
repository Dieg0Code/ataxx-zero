import { useMemo, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AppShell } from "@/widgets/layout/AppShell";
import { useAuth } from "@/app/providers/useAuth";
import { fetchPersistedReplay, type PersistedReplay } from "@/features/match/persistence";
import type { BoardState } from "@/features/match/types";
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

function boardAtPly(replay: PersistedReplay, selectedPly: number): BoardState | null {
  if (replay.moves.length === 0) {
    return null;
  }
  if (selectedPly <= 0) {
    return replay.moves[0].board_before ?? replay.moves[0].board_after;
  }
  const moveIndex = Math.min(selectedPly - 1, replay.moves.length - 1);
  return replay.moves[moveIndex].board_after ?? replay.moves[moveIndex].board_before;
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
  const { accessToken, loading, isAuthenticated } = useAuth();
  const [selectedPly, setSelectedPly] = useState(0);

  const replayQuery = useQuery({
    queryKey: ["profile-game-replay", gameId, accessToken],
    queryFn: () => fetchPersistedReplay(accessToken!, gameId!),
    enabled: Boolean(gameId && accessToken),
  });

  const maxPly = replayQuery.data?.moves.length ?? 0;

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

  const selectedMove = useMemo(() => {
    if (!replayQuery.data || selectedPly <= 0 || selectedPly > replayQuery.data.moves.length) {
      return null;
    }
    return replayQuery.data.moves[selectedPly - 1];
  }, [replayQuery.data, selectedPly]);

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
                      winner: {replayQuery.data.game.winner_side ?? "draw"}
                    </Badge>
                    <Badge variant="default">moves: {replayQuery.data.moves.length}</Badge>
                  </div>

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
                        disabled={selectedPly <= 0}
                        onClick={() => setSelectedPly((prev) => Math.max(0, prev - 1))}
                      >
                        Anterior
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        disabled={selectedPly >= maxPly}
                        onClick={() => setSelectedPly((prev) => Math.min(maxPly, prev + 1))}
                      >
                        Siguiente
                      </Button>
                      <Button type="button" size="sm" variant="secondary" onClick={() => setSelectedPly(0)}>
                        Inicio
                      </Button>
                      <Button type="button" size="sm" variant="secondary" onClick={() => setSelectedPly(maxPly)}>
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
                            <motion.span
                              key={`${rowIdx}:${colIdx}:${selectedPly}:${cell}`}
                              className={`h-4/5 w-4/5 rounded-full border-2 ${cellClass(cell)}`}
                              initial={{ scale: changedCells.has(`${rowIdx}:${colIdx}`) ? 0.72 : 1, opacity: 0.9 }}
                              animate={{ scale: 1, opacity: 1 }}
                              transition={{ type: "spring", stiffness: 280, damping: 22 }}
                            />
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
              {!replayQuery.data || replayQuery.data.moves.length === 0 ? (
                <p className="text-sm text-textDim">Aun no hay jugadas registradas.</p>
              ) : (
                <div className="max-h-[640px] space-y-2 overflow-auto pr-1">
                  {replayQuery.data.moves.map((move, idx) => {
                    const selected = selectedMove?.ply === move.ply && selectedMove.player_side === move.player_side;
                    return (
                      <button
                        key={`${move.ply}-${move.player_side}-${idx}`}
                        type="button"
                        onClick={() => setSelectedPly(idx + 1)}
                        className={`w-full rounded-md border px-3 py-2 text-left text-sm transition ${
                          selected
                            ? "border-primary/70 bg-primary/10"
                            : "border-line/70 bg-black/35 hover:border-primary/40"
                        }`}
                      >
                        <p className="font-medium text-textMain">
                          Ply {move.ply} - {move.player_side}
                        </p>
                        <p className="text-textDim">{renderMoveLabel(move)}</p>
                        <p className="text-xs text-primary">mode: {move.mode}</p>
                      </button>
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
