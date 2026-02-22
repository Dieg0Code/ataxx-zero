import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";
import { motion } from "framer-motion";
import { AppShell } from "@/widgets/layout/AppShell";
import { predictAIMove, type MoveMode } from "@/features/match/api";
import {
  createPersistedGame,
  fetchPersistedReplay,
  storeInferredMove,
  storeManualMove,
  type PersistedReplay,
} from "@/features/match/persistence";
import {
  PLAYER_1,
  PLAYER_2,
  applyMove,
  countPieces,
  createInitialBoard,
  getOutcome,
  getValidMoves,
  hasValidMoves,
  isGameOver,
  normalizeForcedPasses,
  opponent,
} from "@/features/match/rules";
import type { BoardState, Cell, Move } from "@/features/match/types";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";

const HUMAN_PLAYER = PLAYER_1;
const AI_PLAYER = PLAYER_2;
const TOKEN_STORAGE_KEY = "ataxx.apiToken";
const AI_THINK_DELAY_MS = 460;
const AI_PREVIEW_MS = 420;
const INFECTION_STEP_MS = 90;
const INFECTION_BURST_MS = 420;
const INTRO_COUNTDOWN_START = 3;

type OpponentProfile = "model" | "heuristic";
type InfectionMask = Record<string, { oldCell: Cell; revealAt: number }>;
type HeuristicLevel = "easy" | "normal" | "hard";
type InfectionBurst = { key: string; until: number };
type MatchMode = "play" | "spectate";

function cellKey(row: number, col: number): string {
  return `${row}:${col}`;
}

function keyToCell(key: string): { row: number; col: number } {
  const [rRaw, cRaw] = key.split(":");
  return { row: Number(rRaw), col: Number(cRaw) };
}

function cellCenterPercent(index: number): number {
  return ((index + 0.5) / 7) * 100;
}

function winnerLabel(outcome: 1 | -1 | 0 | null): string {
  if (outcome === PLAYER_1) {
    return "Victoria humana: contenciÃ³n exitosa";
  }
  if (outcome === PLAYER_2) {
    return "Derrota: el malware alienÃ­gena domina";
  }
  if (outcome === 0) {
    return "Empate: equilibrio de seÃ±al";
  }
  return "Conflicto de seÃ±al en curso";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function applyInfectionMask(board: BoardState, mask: InfectionMask, nowMs: number): BoardState {
  const grid = board.grid.map((row) => row.slice()) as Cell[][];
  for (const [key, value] of Object.entries(mask)) {
    if (nowMs < value.revealAt) {
      const { row, col } = keyToCell(key);
      if (!Number.isNaN(row) && !Number.isNaN(col)) {
        grid[row][col] = value.oldCell;
      }
    }
  }
  return {
    grid,
    current_player: board.current_player,
    half_moves: board.half_moves,
  };
}

function buildInfectionMask(
  before: BoardState,
  after: BoardState,
  move: Move | null,
  player: 1 | -1,
  nowMs: number,
): InfectionMask {
  if (move === null) {
    return {};
  }
  const destination = cellKey(move.r2, move.c2);
  const infected: Array<{ row: number; col: number; distance: number }> = [];

  for (let r = 0; r < before.grid.length; r += 1) {
    for (let c = 0; c < before.grid[r].length; c += 1) {
      const beforeCell = before.grid[r][c];
      const afterCell = after.grid[r][c];
      const key = cellKey(r, c);
      if (key === destination) {
        continue;
      }
      if (beforeCell === opponent(player) && afterCell === player) {
        infected.push({
          row: r,
          col: c,
          distance: Math.abs(r - move.r2) + Math.abs(c - move.c2),
        });
      }
    }
  }

  infected.sort((a, b) => a.distance - b.distance);
  const mask: InfectionMask = {};
  infected.forEach((cell, idx) => {
    mask[cellKey(cell.row, cell.col)] = {
      oldCell: before.grid[cell.row][cell.col],
      revealAt: nowMs + (idx + 1) * INFECTION_STEP_MS,
    };
  });
  return mask;
}

export function MatchPage(): JSX.Element {
  const [board, setBoard] = useState<BoardState>(() => createInitialBoard());
  const [selected, setSelected] = useState<[number, number] | null>(null);
  const [mode, setMode] = useState<"fast" | "strong">("fast");
  const [thinking, setThinking] = useState(false);
  const [status, setStatus] = useState("Preparando protocolo de contenciÃ³n...");
  const [evalValue, setEvalValue] = useState<number | null>(null);
  const [tokenInput, setTokenInput] = useState<string>(() => localStorage.getItem(TOKEN_STORAGE_KEY) ?? "");
  const [persistedGameId, setPersistedGameId] = useState<string | null>(null);
  const [persistStatus, setPersistStatus] = useState<string>("Modo local (sin persistencia remota).");
  const [replay, setReplay] = useState<PersistedReplay | null>(null);
  const [matchMode, setMatchMode] = useState<MatchMode>("play");
  const [opponentProfile, setOpponentProfile] = useState<OpponentProfile>("model");
  const [heuristicLevel, setHeuristicLevel] = useState<HeuristicLevel>("normal");
  const [p1Profile, setP1Profile] = useState<OpponentProfile>("heuristic");
  const [p1ModelMode, setP1ModelMode] = useState<"fast" | "strong">("fast");
  const [p1HeuristicLevel, setP1HeuristicLevel] = useState<HeuristicLevel>("normal");
  const [previewMove, setPreviewMove] = useState<Move | null>(null);
  const [previewUntil, setPreviewUntil] = useState(0);
  const [infectionMask, setInfectionMask] = useState<InfectionMask>({});
  const [infectionBursts, setInfectionBursts] = useState<InfectionBurst[]>([]);
  const [lastResolvedMove, setLastResolvedMove] = useState<Move | null>(null);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [boardTilt, setBoardTilt] = useState({ x: 0, y: 0 });
  const [introCountdown, setIntroCountdown] = useState(INTRO_COUNTDOWN_START);
  const [showIntro, setShowIntro] = useState(true);

  const counts = useMemo(() => countPieces(board), [board]);
  const totalPieces = counts.p1 + counts.p2;
  const threatLevel = useMemo(
    () => (totalPieces === 0 ? 0 : Math.round((counts.p2 / totalPieces) * 100)),
    [counts.p2, totalPieces],
  );
  const outcome = useMemo(() => getOutcome(board), [board]);
  const gameFinished = useMemo(() => isGameOver(board), [board]);
  const canPersist = tokenInput.trim().length > 0;
  const isSpectate = matchMode === "spectate";

  const sideController = useCallback(
    (side: 1 | -1): OpponentProfile | "human" => {
      if (isSpectate) {
        return side === PLAYER_1 ? p1Profile : opponentProfile;
      }
      return side === HUMAN_PLAYER ? "human" : opponentProfile;
    },
    [isSpectate, opponentProfile, p1Profile],
  );

  const sideMoveMode = useCallback(
    (side: 1 | -1): MoveMode => {
      if (side === PLAYER_1) {
        if (p1Profile === "model") {
          return p1ModelMode;
        }
        return `heuristic_${p1HeuristicLevel}` as MoveMode;
      }
      if (opponentProfile === "model") {
        return mode;
      }
      return `heuristic_${heuristicLevel}` as MoveMode;
    },
    [heuristicLevel, mode, opponentProfile, p1HeuristicLevel, p1ModelMode, p1Profile],
  );

  const allCurrentMoves = useMemo(() => getValidMoves(board, board.current_player), [board]);
  const selectedMoves = useMemo(() => {
    if (selected === null) {
      return [];
    }
    const [r, c] = selected;
    return allCurrentMoves.filter((move) => move.r1 === r && move.c1 === c);
  }, [allCurrentMoves, selected]);
  const selectedTargets = useMemo(
    () => new Set(selectedMoves.map((move) => cellKey(move.r2, move.c2))),
    [selectedMoves],
  );

  const boardForRender = useMemo(
    () => applyInfectionMask(board, infectionMask, nowMs),
    [board, infectionMask, nowMs],
  );

  useEffect(() => {
    const interval = window.setInterval(() => setNowMs(Date.now()), 40);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!showIntro) {
      return;
    }
    const timeout = window.setTimeout(() => {
      if (introCountdown > 0) {
        setIntroCountdown((prev) => prev - 1);
        return;
      }
      setShowIntro(false);
      setStatus(matchMode === "spectate" ? "Â¡Comienza la simulacion IA vs IA!" : "Â¡Comienza la partida! Tu turno, comandante.");
    }, introCountdown === 0 ? 520 : 760);

    return () => window.clearTimeout(timeout);
  }, [introCountdown, matchMode, showIntro]);

  useEffect(() => {
    localStorage.setItem(TOKEN_STORAGE_KEY, tokenInput.trim());
  }, [tokenInput]);

  useEffect(() => {
    if (Object.keys(infectionMask).length === 0) {
      return;
    }
    const nextMask: InfectionMask = {};
    const newlyRevealed: string[] = [];
    for (const [key, value] of Object.entries(infectionMask)) {
      if (nowMs < value.revealAt) {
        nextMask[key] = value;
      } else {
        newlyRevealed.push(key);
      }
    }
    if (newlyRevealed.length > 0) {
      setInfectionBursts((prev) => [
        ...prev,
        ...newlyRevealed.map((key) => ({ key, until: nowMs + INFECTION_BURST_MS })),
      ]);
    }
    if (Object.keys(nextMask).length !== Object.keys(infectionMask).length) {
      setInfectionMask(nextMask);
    }
  }, [infectionMask, nowMs]);

  useEffect(() => {
    setInfectionBursts((prev) => prev.filter((item) => nowMs < item.until));
  }, [nowMs]);

  useEffect(() => {
    if (previewMove !== null && nowMs >= previewUntil) {
      setPreviewMove(null);
    }
  }, [nowMs, previewMove, previewUntil]);

  const resetGame = useCallback(() => {
    setBoard(createInitialBoard());
    setSelected(null);
    setEvalValue(null);
    setStatus("Reiniciando arena... sincronizando seÃ±al.");
    setPersistedGameId(null);
    setPersistStatus("Modo local (sin persistencia remota).");
    setReplay(null);
    setPreviewMove(null);
    setPreviewUntil(0);
    setInfectionMask({});
    setInfectionBursts([]);
    setLastResolvedMove(null);
    setShowIntro(true);
    setIntroCountdown(INTRO_COUNTDOWN_START);
  }, []);

  const applyBoardUpdate = useCallback(
    (next: BoardState, passCount: number) => {
      setBoard(next);
      setSelected(null);
      if (isGameOver(next)) {
        setStatus(`${winnerLabel(getOutcome(next))}. Fin de la operaciÃ³n.`);
        return;
      }
      if (passCount > 0) {
        setStatus(`Auto-pass executed (${passCount}) due to no legal moves.`);
        return;
      }
      const controller = sideController(next.current_player);
      if (controller === "human") {
        setStatus("Turno humano: define tu movimiento.");
      } else {
        setStatus(`Turno IA (${controller}): analizando vector...`);
      }
    },
    [sideController],
  );

  const animateTransition = useCallback(
    (before: BoardState, next: BoardState, move: Move | null, actingPlayer: 1 | -1) => {
      const mask = buildInfectionMask(before, next, move, actingPlayer, Date.now());
      setInfectionMask(mask);
      setLastResolvedMove(move);
      const normalized = normalizeForcedPasses(next);
      applyBoardUpdate(normalized.board, normalized.passes);
    },
    [applyBoardUpdate],
  );

  const commitHumanMove = useCallback(
    async (candidate: Move) => {
      setThinking(true);
      const before = board;
      try {
        let nextBoard: BoardState;
        let appliedMove: Move | null = candidate;
        if (persistedGameId !== null && canPersist) {
          const stored = await storeManualMove(tokenInput.trim(), persistedGameId, board, candidate);
          if (stored.boardAfter === null) {
            throw new Error("Server returned empty board_after while persisting manual move.");
          }
          nextBoard = stored.boardAfter;
          appliedMove = stored.move;
        } else {
          nextBoard = applyMove(board, candidate);
        }
        animateTransition(before, nextBoard, appliedMove, HUMAN_PLAYER);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Invalid move";
        setStatus(message);
      } finally {
        setThinking(false);
      }
    },
    [animateTransition, board, canPersist, persistedGameId, tokenInput],
  );

  const runAIMove = useCallback(async () => {
    setThinking(true);
    const before = board;
    const side = board.current_player as 1 | -1;
    const controller = sideController(side);
    try {
      if (controller === "human") {
        return;
      }
      if (!hasValidMoves(board, side)) {
        const normalized = normalizeForcedPasses(board);
        applyBoardUpdate(normalized.board, normalized.passes);
        return;
      }

      setStatus(`IA (${controller}) calculando jugada...`);
      await sleep(AI_THINK_DELAY_MS);

      const requestMode: MoveMode = sideMoveMode(side);
      const prediction = await predictAIMove(board, requestMode);
      const plannedMove = prediction.move;
      setEvalValue(prediction.value);

      if (plannedMove !== null) {
        setPreviewMove(plannedMove);
        setPreviewUntil(Date.now() + AI_PREVIEW_MS);
        setStatus(`IA (${controller}) confirma ataque...`);
        await sleep(AI_PREVIEW_MS);
      }

      let nextBoard: BoardState;
      let finalMove: Move | null = plannedMove;
      if (!isSpectate && side === AI_PLAYER && persistedGameId !== null && canPersist) {
        if (controller === "model") {
          const stored = await storeInferredMove(tokenInput.trim(), persistedGameId, board, mode);
          if (stored.boardAfter === null) {
            throw new Error("Server returned empty board_after while persisting AI move.");
          }
          nextBoard = stored.boardAfter;
          finalMove = stored.move;
        } else {
          if (plannedMove === null) {
            const normalized = normalizeForcedPasses(board);
            applyBoardUpdate(normalized.board, normalized.passes);
            return;
          }
          const stored = await storeManualMove(tokenInput.trim(), persistedGameId, board, plannedMove);
          if (stored.boardAfter === null) {
            throw new Error("Server returned empty board_after while persisting heuristic move.");
          }
          nextBoard = stored.boardAfter;
          finalMove = stored.move;
        }
      } else {
        nextBoard = applyMove(board, plannedMove);
      }

      setPreviewMove(null);
      setPreviewUntil(0);
      animateTransition(before, nextBoard, finalMove, side);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error desconocido de IA";
      setStatus(`Fallo en jugada IA: ${message}`);
    } finally {
      setThinking(false);
    }
  }, [
    animateTransition,
    applyBoardUpdate,
    board,
    canPersist,
    isSpectate,
    mode,
    persistedGameId,
    sideController,
    sideMoveMode,
    tokenInput,
  ]);

  useEffect(() => {
    if (thinking || isGameOver(board) || showIntro) {
      return;
    }
    if (sideController(board.current_player) === "human") {
      return;
    }
    void runAIMove();
  }, [board, runAIMove, showIntro, sideController, thinking]);

  const startPersistedSession = useCallback(async () => {
    if (isSpectate) {
      setPersistStatus("Persistencia deshabilitada en modo espectador.");
      return;
    }
    if (!canPersist) {
      setPersistStatus("Primero configura un Bearer token.");
      return;
    }
    setThinking(true);
    try {
      const gameId = await createPersistedGame(tokenInput.trim());
      setPersistedGameId(gameId);
      setPersistStatus(`Persistencia activa: ${gameId.slice(0, 8)}...`);
      setReplay(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "No se pudo crear la partida persistida";
      setPersistStatus(`Persist failed: ${message}`);
    } finally {
      setThinking(false);
    }
  }, [canPersist, isSpectate, tokenInput]);

  const copyGameId = useCallback(async () => {
    if (persistedGameId === null) {
      return;
    }
    try {
      await navigator.clipboard.writeText(persistedGameId);
      setPersistStatus("ID de partida copiado al portapapeles.");
    } catch {
      setPersistStatus("No se pudo copiar el ID de partida.");
    }
  }, [persistedGameId]);

  const refreshReplay = useCallback(async () => {
    if (!canPersist || persistedGameId === null) {
      return;
    }
    setThinking(true);
    try {
      const payload = await fetchPersistedReplay(tokenInput.trim(), persistedGameId);
      setReplay(payload);
      setPersistStatus(`Replay sincronizado. Jugadas: ${payload.moves.length}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "No se pudo obtener el replay";
      setPersistStatus(`Replay error: ${message}`);
    } finally {
      setThinking(false);
    }
  }, [canPersist, persistedGameId, tokenInput]);

  const onCellClick = useCallback(
    (row: number, col: number) => {
      if (isSpectate || thinking || isGameOver(board) || board.current_player !== HUMAN_PLAYER || showIntro) {
        return;
      }
      const cell = board.grid[row][col];
      if (cell === HUMAN_PLAYER) {
        setSelected([row, col]);
        return;
      }
      if (selected === null) {
        return;
      }
      const candidate = selectedMoves.find((move) => move.r2 === row && move.c2 === col);
      if (!candidate) {
        if (selected[0] === row && selected[1] === col) {
          setSelected(null);
        }
        return;
      }
      void commitHumanMove(candidate);
    },
    [board, commitHumanMove, isSpectate, selected, selectedMoves, showIntro, thinking],
  );

  const previewOrigin = previewMove ? keyToCell(cellKey(previewMove.r1, previewMove.c1)) : null;
  const previewTarget = previewMove ? keyToCell(cellKey(previewMove.r2, previewMove.c2)) : null;
  const lastOrigin = lastResolvedMove ? keyToCell(cellKey(lastResolvedMove.r1, lastResolvedMove.c1)) : null;
  const lastTarget = lastResolvedMove ? keyToCell(cellKey(lastResolvedMove.r2, lastResolvedMove.c2)) : null;
  const previewVector =
    previewOrigin !== null && previewTarget !== null
      ? {
          x1: cellCenterPercent(previewOrigin.col),
          y1: cellCenterPercent(previewOrigin.row),
          x2: cellCenterPercent(previewTarget.col),
          y2: cellCenterPercent(previewTarget.row),
        }
      : null;

  const onBoardMouseMove = useCallback((event: MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const nx = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    const ny = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
    setBoardTilt({
      x: -(ny * 2.1),
      y: nx * 2.1,
    });
  }, []);

  const onBoardMouseLeave = useCallback(() => {
    setBoardTilt({ x: 0, y: 0 });
  }, []);

  return (
    <AppShell>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Card className="overflow-hidden border-zinc-800/90 bg-gradient-to-b from-zinc-950/95 to-black/95">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <div>
                <Badge className="mb-2 border-lime-400/35 bg-lime-300/10 text-lime-200" variant="default">
                  EXTRATERRESTRIAL SIGNAL // CONTAINMENT PROTOCOL
                </Badge>
                <CardTitle>Malware Arena</CardTitle>
                <CardDescription>{isSpectate ? `Simulacion IA vs IA (P1 ${p1Profile} vs P2 ${opponentProfile}).` : `Comando humano vs inteligencia alienigena (${opponentProfile}).`}</CardDescription>
              </div>
              <Button variant="secondary" size="sm" onClick={resetGame}>
                Reset
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <motion.div
              className="relative aspect-square max-w-[620px] [perspective:1000px]"
              onMouseMove={onBoardMouseMove}
              onMouseLeave={onBoardMouseLeave}
            >
              <motion.div
                className="relative grid h-full grid-cols-7 gap-1 rounded-xl border border-zinc-800 bg-[radial-gradient(circle_at_50%_-20%,rgba(132,204,22,0.22),rgba(0,0,0,0))] p-2"
                animate={{
                  rotateX: boardTilt.x,
                  rotateY: boardTilt.y,
                  boxShadow: [
                    "0 0 0px rgba(132,204,22,0.0)",
                    "0 0 24px rgba(132,204,22,0.14)",
                    "0 0 0px rgba(132,204,22,0.0)",
                  ],
                }}
                transition={{
                  rotateX: { type: "spring", stiffness: 180, damping: 22, mass: 0.7 },
                  rotateY: { type: "spring", stiffness: 180, damping: 22, mass: 0.7 },
                  boxShadow: { duration: 4.2, repeat: Infinity, ease: "easeInOut" },
                }}
              >
                <motion.div
                  className="pointer-events-none absolute inset-0 z-20 rounded-xl opacity-35"
                  style={{
                    background:
                      "repeating-linear-gradient(180deg, rgba(255,255,255,0.08) 0px, rgba(255,255,255,0.08) 1px, rgba(0,0,0,0) 2px, rgba(0,0,0,0) 4px)",
                  }}
                  animate={{ backgroundPositionY: ["0px", "120px"] }}
                  transition={{ duration: 4.6, ease: "linear", repeat: Infinity }}
                />
                <motion.div
                  className="pointer-events-none absolute left-1/2 top-1/2 z-20 h-[55%] w-[55%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-lime-200/20"
                  animate={{ scale: [0.94, 1.03, 0.94], opacity: [0.2, 0.42, 0.2] }}
                  transition={{ duration: 3.6, ease: "easeInOut", repeat: Infinity }}
                />
                <motion.div
                  className="pointer-events-none absolute inset-0 z-20 rounded-xl"
                  style={{
                    background:
                      "radial-gradient(circle at 50% 45%, rgba(132,204,22,0.08), rgba(0,0,0,0.0) 42%, rgba(0,0,0,0.32) 100%)",
                  }}
                  animate={{ opacity: [0.75, 0.9, 0.75] }}
                  transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
                />
                {showIntro && (
                  <motion.div
                    className="pointer-events-none absolute inset-0 z-40 flex flex-col items-center justify-center bg-black/58 text-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.25 }}
                  >
                    <p className="mb-2 text-xs uppercase tracking-[0.22em] text-zinc-400">Sincronizando protocolo</p>
                    <motion.p
                      key={`intro-${introCountdown}`}
                      className="text-6xl font-black text-lime-200 drop-shadow-[0_0_18px_rgba(132,204,22,0.65)]"
                      initial={{ scale: 0.72, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 1.2, opacity: 0 }}
                      transition={{ duration: 0.3, ease: "easeOut" }}
                    >
                      {introCountdown > 0 ? introCountdown : "Â¡LUCHA!"}
                    </motion.p>
                  </motion.div>
                )}
                {gameFinished && !showIntro && (
                  <motion.div
                    className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center bg-black/45 px-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="rounded-xl border border-zinc-300/35 bg-zinc-950/90 px-5 py-4 text-center">
                      <p className="text-xs uppercase tracking-[0.2em] text-zinc-400">Resultado</p>
                      <p className="mt-1 text-lg font-semibold text-zinc-100">{winnerLabel(outcome)}</p>
                    </div>
                  </motion.div>
                )}

              {previewVector !== null && (
                <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                  <motion.line
                    x1={previewVector.x1}
                    y1={previewVector.y1}
                    x2={previewVector.x2}
                    y2={previewVector.y2}
                    stroke="rgba(132,204,22,0.85)"
                    strokeWidth="0.55"
                    strokeDasharray="2 2"
                    initial={{ pathLength: 0, opacity: 0.2, strokeDashoffset: 8 }}
                    animate={{ pathLength: 1, opacity: 1, strokeDashoffset: 0 }}
                    transition={{ duration: 0.28, ease: "easeOut" }}
                  />
                  <motion.line
                    x1={previewVector.x1}
                    y1={previewVector.y1}
                    x2={previewVector.x2}
                    y2={previewVector.y2}
                    stroke="rgba(255,255,255,0.55)"
                    strokeWidth="0.2"
                    initial={{ pathLength: 0, opacity: 0.2 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ duration: 0.28, ease: "easeOut" }}
                  />
                </svg>
              )}

              {infectionBursts.map((burst) => {
                const { row, col } = keyToCell(burst.key);
                return (
                  <motion.span
                    key={`${burst.key}-${burst.until}`}
                    className="pointer-events-none absolute z-30 h-6 w-6 rounded-full border border-lime-300"
                    style={{
                      top: `${cellCenterPercent(row)}%`,
                      left: `${cellCenterPercent(col)}%`,
                      translate: "-50% -50%",
                    }}
                    initial={{ scale: 0.2, opacity: 0.9 }}
                    animate={{ scale: 2.4, opacity: 0 }}
                    transition={{ duration: INFECTION_BURST_MS / 1000, ease: "easeOut" }}
                  />
                );
              })}

                {boardForRender.grid.map((row, r) =>
                row.map((cell, c) => {
                  const key = cellKey(r, c);
                  const isSelected = selected !== null && selected[0] === r && selected[1] === c;
                  const isTarget = selectedTargets.has(key);
                  const canPick = !isSpectate && board.current_player === HUMAN_PLAYER && cell === HUMAN_PLAYER;
                  const isPreviewOrigin = previewMove !== null && previewMove.r1 === r && previewMove.c1 === c;
                  const isPreviewTarget = previewMove !== null && previewMove.r2 === r && previewMove.c2 === c;
                  const isRecentOrigin = lastOrigin !== null && lastOrigin.row === r && lastOrigin.col === c;
                  const isRecentTarget = lastTarget !== null && lastTarget.row === r && lastTarget.col === c;

                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onCellClick(r, c)}
                      disabled={isSpectate || thinking || isGameOver(board)}
                      className={`relative z-10 flex aspect-square items-center justify-center rounded border transition ${
                        isPreviewTarget
                          ? "border-lime-300/90 bg-lime-300/15"
                          : isSelected
                            ? "border-zinc-300 bg-zinc-200/15"
                            : isTarget
                              ? "border-zinc-500 bg-zinc-200/10"
                              : canPick
                                ? "border-zinc-700 bg-zinc-900/60 hover:border-zinc-500"
                                : "border-zinc-800 bg-zinc-950/60"
                      }`}
                    >
                      {cell !== 0 && (
                        <motion.span
                          layout
                          className={`h-4/5 w-4/5 rounded-full border-2 transition ${
                            cell === PLAYER_1
                              ? "border-zinc-200 bg-zinc-100 shadow-[0_0_14px_rgba(255,255,255,0.35)]"
                              : "border-lime-300 bg-lime-400 shadow-[0_0_16px_rgba(132,204,22,0.45)]"
                          } ${
                            isPreviewOrigin || isRecentTarget ? "scale-110" : ""
                          }`}
                          initial={{ scale: 0.8, opacity: 0.8 }}
                          animate={{
                            scale: isPreviewOrigin ? 1.1 : 1,
                            opacity: 1,
                            y: isPreviewOrigin ? [-1, 1, -1] : 0,
                            boxShadow:
                              cell === PLAYER_2
                                ? [
                                    "0 0 8px rgba(132,204,22,0.32)",
                                    "0 0 18px rgba(132,204,22,0.58)",
                                    "0 0 8px rgba(132,204,22,0.32)",
                                  ]
                                : [
                                    "0 0 7px rgba(255,255,255,0.22)",
                                    "0 0 13px rgba(255,255,255,0.36)",
                                    "0 0 7px rgba(255,255,255,0.22)",
                                  ],
                          }}
                          transition={{
                            scale: { type: "spring", stiffness: 360, damping: 24 },
                            opacity: { duration: 0.22 },
                            y: { duration: 1.2, repeat: Infinity, ease: "easeInOut" },
                            boxShadow: { duration: 1.8, repeat: Infinity, ease: "easeInOut" },
                          }}
                        />
                      )}
                      {isTarget && <span className="absolute h-2.5 w-2.5 rounded-full bg-zinc-200" />}
                      {(isRecentOrigin || isRecentTarget) && (
                        <motion.span
                          className="absolute inset-1 rounded border border-zinc-300/50"
                          animate={{ opacity: [0.3, 0.9, 0.3] }}
                          transition={{ duration: 0.85, repeat: 2, ease: "easeInOut" }}
                        />
                      )}
                    </button>
                  );
                  }),
                )}
              </motion.div>
            </motion.div>
          </CardContent>
        </Card>

        <Card className="border-zinc-800/90 bg-black/90">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Threat Console</CardTitle>
            <CardDescription>Consola tÃ¡ctica y telemetrÃ­a del malware.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1 text-sm text-zinc-300">
              <p>
                Turn:{" "}
                <span className="font-semibold text-zinc-100">
                  {board.current_player === PLAYER_1 ? sideController(PLAYER_1) === "human" ? "Equipo humano" : `IA P1 (${sideController(PLAYER_1)})` : `IA P2 (${sideController(PLAYER_2)})`}
                </span>
              </p>
              <p>
                Estado: <span className="font-semibold text-zinc-100">{winnerLabel(outcome)}</span>
              </p>
              <p>
                Fichas: <span className="font-semibold text-zinc-100">{counts.p1}</span> vs{" "}
                <span className="font-semibold text-lime-300">{counts.p2}</span>
              </p>
              <p>Medios turnos: {board.half_moves}</p>
              <p>EvaluaciÃ³n IA: {evalValue === null ? "-" : evalValue.toFixed(3)}</p>
            </div>

            <div>
              <div className="mb-1 flex items-center justify-between text-xs text-zinc-400">
                <span>Nivel de amenaza malware</span>
                <span>{threatLevel}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-900">
                <motion.div
                  className="h-full bg-gradient-to-r from-lime-500 to-lime-300"
                  initial={false}
                  animate={{ width: `${threatLevel}%` }}
                  transition={{ duration: 0.35, ease: "easeOut" }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div>
                <label htmlFor="match-mode" className="mb-1 block text-xs text-zinc-400">
                  Modo de partida
                </label>
                <select
                  id="match-mode"
                  value={matchMode}
                  onChange={(event) => setMatchMode(event.target.value as MatchMode)}
                  className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                  disabled={thinking}
                >
                  <option value="play">Jugar (Humano vs IA)</option>
                  <option value="spectate">Espectar (IA vs IA)</option>
                </select>
              </div>
              {isSpectate ? (
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label htmlFor="p1-opponent" className="mb-1 block text-xs text-zinc-400">
                      P1 agente
                    </label>
                    <select
                      id="p1-opponent"
                      value={p1Profile}
                      onChange={(event) => setP1Profile(event.target.value as OpponentProfile)}
                      className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                      disabled={thinking}
                    >
                      <option value="model">modelo</option>
                      <option value="heuristic">heuristica</option>
                    </select>
                  </div>
                  {p1Profile === "model" ? (
                    <div>
                      <label htmlFor="p1-mode" className="mb-1 block text-xs text-zinc-400">
                        P1 modo
                      </label>
                      <select
                        id="p1-mode"
                        value={p1ModelMode}
                        onChange={(event) => setP1ModelMode(event.target.value as "fast" | "strong")}
                        className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                        disabled={thinking}
                      >
                        <option value="fast">rapido</option>
                        <option value="strong">fuerte</option>
                      </select>
                    </div>
                  ) : (
                    <div>
                      <label htmlFor="p1-level" className="mb-1 block text-xs text-zinc-400">
                        P1 nivel
                      </label>
                      <select
                        id="p1-level"
                        value={p1HeuristicLevel}
                        onChange={(event) => setP1HeuristicLevel(event.target.value as HeuristicLevel)}
                        className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                        disabled={thinking}
                      >
                        <option value="easy">easy</option>
                        <option value="normal">normal</option>
                        <option value="hard">hard</option>
                      </select>
                    </div>
                  )}
                </div>
              ) : null}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label htmlFor="opponent" className="mb-1 block text-xs text-zinc-400">
                  {isSpectate ? "P2 agente" : "Oponente"}
                </label>
                <select
                  id="opponent"
                  value={opponentProfile}
                  onChange={(event) => setOpponentProfile(event.target.value as OpponentProfile)}
                  className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                  disabled={thinking}
                >
                  <option value="model">modelo</option>
                  <option value="heuristic">heurÃ­stica</option>
                </select>
              </div>

              {opponentProfile === "model" ? (
                <div>
                  <label htmlFor="mode" className="mb-1 block text-xs text-zinc-400">
                    Modo del modelo
                  </label>
                  <select
                    id="mode"
                    value={mode}
                    onChange={(event) => setMode(event.target.value as "fast" | "strong")}
                    className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                    disabled={thinking}
                  >
                    <option value="fast">rÃ¡pido</option>
                    <option value="strong">fuerte</option>
                  </select>
                </div>
              ) : (
                <div>
                  <label htmlFor="heuristic-level" className="mb-1 block text-xs text-zinc-400">
                    Nivel heurÃ­stico
                  </label>
                  <select
                    id="heuristic-level"
                    value={heuristicLevel}
                    onChange={(event) => setHeuristicLevel(event.target.value as HeuristicLevel)}
                    className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100"
                    disabled={thinking}
                  >
                    <option value="easy">easy</option>
                    <option value="normal">normal</option>
                    <option value="hard">hard</option>
                  </select>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <label htmlFor="token" className="block text-xs text-zinc-400">
                Token Bearer API
              </label>
              <Input
                id="token"
                type="password"
                value={tokenInput}
                onChange={(event) => setTokenInput(event.target.value)}
                placeholder="Habilitar persistencia remota"
                disabled={thinking}
              />
              <Button
                type="button"
                onClick={() => void startPersistedSession()}
                disabled={isSpectate || thinking || !canPersist || persistedGameId !== null}
                variant="secondary"
                size="sm"
                className="w-full"
              >
                {persistedGameId === null ? "Iniciar partida persistida" : "Partida persistida activa"}
              </Button>
            </div>

            {persistedGameId !== null && (
              <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2 text-[11px] text-zinc-300">
                <p className="break-all">
                  game_id: <span className="text-zinc-100">{persistedGameId}</span>
                </p>
                <div className="mt-2 flex gap-2">
                  <Button type="button" size="sm" variant="secondary" onClick={() => void copyGameId()}>
                    Copiar ID
                  </Button>
                  <Button type="button" size="sm" variant="secondary" onClick={() => void refreshReplay()}>
                    Actualizar replay
                  </Button>
                </div>
              </div>
            )}

            <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2 text-xs text-zinc-300">
              {thinking ? `IA activa (${sideController(board.current_player)}) pensando...` : status}
            </div>
            <p className="text-[11px] text-zinc-500">{persistStatus}</p>
            {replay !== null && (
              <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2 text-[11px] text-zinc-300">
                <p>
                  Estado replay: <span className="text-zinc-100">{replay.game.status}</span>
                </p>
                <p>
                  Ganador: <span className="text-zinc-100">{replay.game.winner_side ?? "-"}</span>
                </p>
                <p>
                  Total jugadas: <span className="text-zinc-100">{replay.moves.length}</span>
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}



