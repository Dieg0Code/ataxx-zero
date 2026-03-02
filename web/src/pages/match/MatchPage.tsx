import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Bot,
  Brain,
  Crown,
  Gauge,
  RotateCcw,
  X,
  Search,
  ShieldAlert,
  LogOut,
  Swords,
  Timer,
  User,
} from "lucide-react";
import { AppShell } from "@/widgets/layout/AppShell";
import { useAuth } from "@/app/providers/useAuth";
import {
  fetchPublicPlayers,
  type PlayableBot,
  type PublicPlayer,
} from "@/features/identity/api";
import { fetchActiveSeason, fetchRatingEvents, fetchUserRating } from "@/features/ranking/api";
import { predictAIMove, type MoveMode } from "@/features/match/api";
import {
  createPersistedGame,
  deletePersistedGame,
  fetchPersistedGameSummary,
  fetchPersistedReplay,
  openPersistedGameSocket,
  type PersistedGameWsEvent,
  storeManualMove,
  type PersistedMoveMode,
  type PersistedReplay,
} from "@/features/match/persistence";
import { createHumanInvitation, fetchInvitationGame } from "@/features/matches/api";
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

const HUMAN_PLAYER = PLAYER_1;
const AI_THINK_DELAY_MS = 460;
const AI_PREVIEW_MS = 420;
const INFECTION_STEP_MS = 90;
const INFECTION_BURST_MS = 420;
const INTRO_COUNTDOWN_START = 3;
const HOVER_SFX_MIN_GAP_MS = 120;
const P2_MOVE_SFX_MIN_GAP_MS = 70;
const PERSIST_MAX_RETRIES = 3;
const PERSIST_BASE_BACKOFF_MS = 200;
const PERSIST_SNAPSHOT_KEY = "ataxx.persist.snapshot.v1";
const MATCHMAKING_PRESET_KEY = "ataxx.matchmaking.preset.v1";
const MATCHMAKING_MATCH_KEY = "ataxx.matchmaking.match.v1";
const MATCHMAKING_AUTOQUEUE_KEY = "ataxx.matchmaking.autoqueue.v1";
const GUEST_PROFILE_KEY = "ataxx.guest.profile.v1";
const SFX = {
  uiClick: "/sfx/ui_click.ogg",
  uiHover: "/sfx/start.ogg",
  start: "/sfx/start.ogg",
  moveLand: "/sfx/move_preview.ogg",
  infect: "/sfx/infect.ogg",
  win: "/sfx/win.ogg",
  lose: "/sfx/lose.ogg",
  resultReveal: "/sfx/move_preview.ogg",
  queueDeploy: "/sfx/queue_accept.ogg",
} as const;

const panelSectionVariants: Variants = {
  hidden: { opacity: 0, y: 8 },
  show: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.32, ease: "easeOut" as const, delay },
  }),
};

type OpponentProfile = "model" | "heuristic";
type InfectionMask = Record<string, { oldCell: Cell; revealAt: number }>;
type HeuristicLevel = "easy" | "normal" | "hard";
type InfectionBurst = { key: string; until: number };
type MatchMode = "play" | "spectate";
type ControllerKind = "human" | "remote_human" | "model" | "heuristic";
type PendingPersistOperation = {
  gameId: string;
  beforeBoard: BoardState;
  move: Move;
  mode: PersistedMoveMode;
  actorLabel: string;
};
type MatchmakingPreset = {
  opponentProfile?: OpponentProfile;
  mode?: "fast" | "strong";
  heuristicLevel?: HeuristicLevel;
  guestMode?: boolean;
  guestUsername?: string;
  guestRivalName?: string;
  createdAt?: number;
};
type MatchmakingMatch = {
  gameId?: string;
  matchedWith?: "human" | "bot";
  createdAt?: number;
};
type OutgoingInvitation = {
  gameId: string;
  opponentUserId: string;
  opponentUsername: string;
  rated: boolean;
};
type RatingBaseline = {
  seasonId: string;
  rating: number;
  lp: number;
  league: string;
  division: string;
  gamesPlayed: number;
};

function playerOptionLabel(player: PublicPlayer): string {
  if (player.is_bot) {
    if (player.agent_type === "model") {
      return `${player.username} [bot:model:${player.model_mode ?? "fast"}]`;
    }
    return `${player.username} [bot:heuristic:${player.heuristic_level ?? "normal"}]`;
  }
  return `${player.username} [humano]`;
}

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
    return "Victoria de P1";
  }
  if (outcome === PLAYER_2) {
    return "Victoria de P2";
  }
  if (outcome === 0) {
    return "Empate tecnico";
  }
  return "Partida en curso";
}

function formatSignedDelta(value: number): string {
  const rounded = Number.isInteger(value) ? value.toString() : value.toFixed(1);
  return value > 0 ? `+${rounded}` : rounded;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function loadGuestProfileUsername(): string | null {
  try {
    const raw = localStorage.getItem(GUEST_PROFILE_KEY);
    if (raw === null) {
      return null;
    }
    const parsed = JSON.parse(raw) as { username?: string };
    if (typeof parsed.username !== "string") {
      return null;
    }
    const cleaned = parsed.username.trim();
    return cleaned.length > 0 ? cleaned : null;
  } catch {
    return null;
  }
}

function playSfx(path: string, volume = 0.3): void {
  try {
    const audio = new Audio(path);
    audio.volume = Math.max(0, Math.min(1, volume));
    const playResult = audio.play();
    if (playResult && typeof playResult.catch === "function") {
      void playResult.catch(() => {});
    }
  } catch {
    // ignore audio runtime errors in unsupported environments
  }
}

function isFatalPersistenceError(error: unknown): boolean {
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  return (
    message.includes("game not found") ||
    message.includes("not allowed") ||
    message.includes("unauthorized") ||
    message.includes("missing authorization token") ||
    message.includes("http 401") ||
    message.includes("http 403") ||
    message.includes("http 404")
  );
}

async function withRetry<T>(
  fn: () => Promise<T>,
  options?: { attempts?: number; baseDelayMs?: number },
): Promise<T> {
  const attempts = Math.max(1, options?.attempts ?? 3);
  const baseDelayMs = Math.max(0, options?.baseDelayMs ?? 180);
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await fn();
    } catch (error: unknown) {
      lastError = error;
      if (attempt >= attempts) {
        break;
      }
      await sleep(baseDelayMs * attempt);
    }
  }
  throw lastError;
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
  const { user, isAuthenticated, accessToken, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [board, setBoard] = useState<BoardState>(() => createInitialBoard());
  const [selected, setSelected] = useState<[number, number] | null>(null);
  const [mode, setMode] = useState<"fast" | "strong">("fast");
  const [thinking, setThinking] = useState(false);
  const [persisting, setPersisting] = useState(false);
  const [status, setStatus] = useState("Configura la partida y pulsa Iniciar.");
  const [evalValue, setEvalValue] = useState<number | null>(null);
  const [persistedGameId, setPersistedGameId] = useState<string | null>(null);
  const [persistStatus, setPersistStatus] = useState<string>("Modo local (sin persistencia remota).");
  const [persistError, setPersistError] = useState<string | null>(null);
  const [retryingPersist, setRetryingPersist] = useState(false);
  const [pendingPersistCount, setPendingPersistCount] = useState(0);
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
  const [showIntro, setShowIntro] = useState(false);
  const [matchStarted, setMatchStarted] = useState(false);
  const [matchStartMs, setMatchStartMs] = useState<number | null>(null);
  const [matchEndMs, setMatchEndMs] = useState<number | null>(null);
  const [resolvedMoves, setResolvedMoves] = useState(0);
  const [queueNotice, setQueueNotice] = useState<string | null>(null);
  const [autoQueueStart, setAutoQueueStart] = useState(false);
  const [queueRanked, setQueueRanked] = useState(false);
  const [queuedGameId, setQueuedGameId] = useState<string | null>(null);
  const [guestQueueMode, setGuestQueueMode] = useState(false);
  const [guestUsername, setGuestUsername] = useState<string | null>(() => loadGuestProfileUsername());
  const [guestRivalName, setGuestRivalName] = useState<string | null>(null);
  const [queueRedirecting, setQueueRedirecting] = useState(false);
  const [finishLpDelta, setFinishLpDelta] = useState<number | null>(null);
  const [finishRatingDelta, setFinishRatingDelta] = useState<number | null>(null);
  const [finishTransitionType, setFinishTransitionType] = useState<"promotion" | "demotion" | "stable" | null>(null);
  const [loadingFinishRewards, setLoadingFinishRewards] = useState(false);
  const [animatedLpDelta, setAnimatedLpDelta] = useState<number | null>(null);
  const [animatedRatingDelta, setAnimatedRatingDelta] = useState<number | null>(null);
  const [showExitConfirm, setShowExitConfirm] = useState(false);
  const [exitingMatch, setExitingMatch] = useState(false);
  const [pendingNavigationPath, setPendingNavigationPath] = useState<string | null>(null);
  const [pendingLogout, setPendingLogout] = useState(false);
  const [queueArenaP1Elo, setQueueArenaP1Elo] = useState<number | null>(null);
  const [queueArenaP2Elo, setQueueArenaP2Elo] = useState<number | null>(null);
  const [queueArenaLoading, setQueueArenaLoading] = useState(false);
  const [localPlayerSide, setLocalPlayerSide] = useState<1 | -1>(PLAYER_1);
  const [queuePlayerAgents, setQueuePlayerAgents] = useState<{
    p1: "human" | "model" | "heuristic";
    p2: "human" | "model" | "heuristic";
  } | null>(null);
  const [outgoingInvitation, setOutgoingInvitation] = useState<OutgoingInvitation | null>(null);
  const [ratingBaseline, setRatingBaseline] = useState<RatingBaseline | null>(null);
  const [publicPlayers, setPublicPlayers] = useState<PublicPlayer[]>([]);
  const [botAccounts, setBotAccounts] = useState<PlayableBot[]>([]);
  const [rivalPickerOpen, setRivalPickerOpen] = useState(false);
  const [rivalPickerTarget, setRivalPickerTarget] = useState<"p1" | "p2">("p2");
  const [rivalPickerQuery, setRivalPickerQuery] = useState("");
  const [selectedP1BotId, setSelectedP1BotId] = useState<string>("");
  const [selectedP2BotId, setSelectedP2BotId] = useState<string>("");
  const lastHoverSfxAtRef = useRef(0);
  const lastP2MoveSfxAtRef = useRef(0);
  const gameplayWsRef = useRef<WebSocket | null>(null);
  const lastWsPlyRef = useRef(-1);
  const persistQueueRef = useRef<Promise<void>>(Promise.resolve());
  const failedPersistOpsRef = useRef<PendingPersistOperation[]>([]);
  const unmountCleanupTriggeredRef = useRef(false);
  const unmountCleanupStateRef = useRef<{
    accessToken: string | null;
    persistedGameId: string | null;
    matchStarted: boolean;
    gameFinished: boolean;
    exitingMatch: boolean;
  }>({
    accessToken: null,
    persistedGameId: null,
    matchStarted: false,
    gameFinished: false,
    exitingMatch: false,
  });
  const lastAccessTokenRef = useRef<string | null>(accessToken);
  const resultRevealPlayedRef = useRef(false);
  const emitFlash = useCallback(
    (message: string, tone: "success" | "warning" | "error" | "info") => {
      navigate(`${location.pathname}${location.search}${location.hash}`, {
        replace: true,
        state: { flash: { message, tone } },
      });
    },
    [location.hash, location.pathname, location.search, navigate],
  );

  useEffect(() => {
    if (accessToken !== null) {
      lastAccessTokenRef.current = accessToken;
    }
  }, [accessToken]);

  useEffect(() => {
    unmountCleanupStateRef.current = {
      accessToken,
      persistedGameId,
      matchStarted,
      gameFinished: isGameOver(board),
      exitingMatch,
    };
  }, [accessToken, board, exitingMatch, matchStarted, persistedGameId]);

  const counts = useMemo(() => countPieces(board), [board]);
  const totalPieces = counts.p1 + counts.p2;
  const threatLevel = useMemo(
    () => (totalPieces === 0 ? 0 : Math.round((counts.p2 / totalPieces) * 100)),
    [counts.p2, totalPieces],
  );
  const outcome = useMemo(() => getOutcome(board), [board]);
  const gameFinished = useMemo(() => isGameOver(board), [board]);
  const canPersist = Boolean(accessToken);
  const isSpectate = matchMode === "spectate";
  const interactionLocked = thinking || persisting;
  const hasBlockingRemoteMatch = matchStarted && !gameFinished && persistedGameId !== null && !exitingMatch;

  const sideController = useCallback(
    (side: 1 | -1): ControllerKind => {
      if (isSpectate) {
        return side === PLAYER_1 ? p1Profile : opponentProfile;
      }
      if (queuedGameId !== null && queuePlayerAgents !== null) {
        if (side === localPlayerSide) {
          return "human";
        }
        const remoteAgent = side === PLAYER_1 ? queuePlayerAgents.p1 : queuePlayerAgents.p2;
        return remoteAgent === "human" ? "remote_human" : remoteAgent;
      }
      return side === HUMAN_PLAYER ? "human" : opponentProfile;
    },
    [isSpectate, localPlayerSide, opponentProfile, p1Profile, queuePlayerAgents, queuedGameId],
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
  const p1SummaryLabel = isSpectate
    ? `P1 (${sideController(PLAYER_1)})`
    : sideController(PLAYER_1) === "human"
      ? "Humano"
      : sideController(PLAYER_1) === "remote_human"
        ? "Rival"
        : `IA (${sideController(PLAYER_1)})`;
  const p2SummaryLabel = isSpectate
    ? `P2 (${sideController(PLAYER_2)})`
    : sideController(PLAYER_2) === "human"
      ? "Humano"
      : sideController(PLAYER_2) === "remote_human"
        ? "Rival"
        : `IA (${sideController(PLAYER_2)})`;
  const currentTurnController = sideController(board.current_player);
  const currentTurnLabel = useMemo(() => {
    if (board.current_player === PLAYER_1) {
      if (currentTurnController === "human") {
        return "Tu turno (P1)";
      }
      if (currentTurnController === "remote_human") {
        return "Turno rival (P1)";
      }
      return `IA P1 (${currentTurnController})`;
    }
    if (currentTurnController === "human") {
      return "Tu turno (P2)";
    }
    if (currentTurnController === "remote_human") {
      return "Turno rival (P2)";
    }
    return `IA P2 (${currentTurnController})`;
  }, [board.current_player, currentTurnController]);
  const matchDurationSeconds = useMemo(() => {
    if (matchStartMs === null) {
      return 0;
    }
    const end = matchEndMs ?? nowMs;
    return Math.max(0, Math.round((end - matchStartMs) / 1000));
  }, [matchEndMs, matchStartMs, nowMs]);
  const matchPace = useMemo(() => {
    if (!matchStarted || matchDurationSeconds <= 0) {
      return "-";
    }
    return `${((resolvedMoves * 60) / matchDurationSeconds).toFixed(1)} ply/min`;
  }, [matchDurationSeconds, matchStarted, resolvedMoves]);
  const controlDelta = counts.p1 - counts.p2;
  const controlDeltaLabel = `${controlDelta >= 0 ? "+" : ""}${controlDelta}`;
  const matchTypeLabel = queueRanked ? "Ranked" : "Casual";
  const isQueueMatchView = queuedGameId !== null || guestQueueMode;
  const isHumanVsHumanMatch =
    !isSpectate &&
    queuedGameId !== null &&
    queuePlayerAgents?.p1 === "human" &&
    queuePlayerAgents?.p2 === "human";
  const modeLabel = matchMode === "spectate" ? "Simulacion" : isHumanVsHumanMatch ? "Humano vs humano" : "Humano vs IA";
  const playerDisplayName = user?.username ?? guestUsername ?? "guest";
  const selectedP1Player = publicPlayers.find((player) => player.user_id === selectedP1BotId) ?? null;
  const selectedP2Player = publicPlayers.find((player) => player.user_id === selectedP2BotId) ?? null;
  const selectedP1Bot = botAccounts.find((bot) => bot.user_id === selectedP1BotId) ?? null;
  const selectedP2Bot = botAccounts.find((bot) => bot.user_id === selectedP2BotId) ?? null;
  const rivalName = selectedP2Player?.username ?? guestRivalName ?? "rival";
  const p1DisplayName = isSpectate
    ? selectedP1Player?.username ?? "p1"
    : localPlayerSide === PLAYER_1
      ? `@${playerDisplayName}`
      : rivalName;
  const p2DisplayName = isSpectate
    ? selectedP2Player?.username ?? "p2"
    : localPlayerSide === PLAYER_2
      ? `@${playerDisplayName}`
      : rivalName;
  const playerLabel = isSpectate
    ? `P1 | ${selectedP1Player?.username ?? "jugador"}`
    : `@${playerDisplayName}`;
  const rivalDisplayLabel = isSpectate
    ? `P2 | ${selectedP2Player?.username ?? guestRivalName ?? "jugador"}`
    : `${selectedP2Player?.username ?? guestRivalName ?? "jugador"}`;
  const winnerName = outcome === PLAYER_1 ? p1DisplayName : outcome === PLAYER_2 ? p2DisplayName : null;
  const winnerHighlightClass =
    outcome === PLAYER_2
      ? "border-lime-300/55 bg-lime-400/12 text-lime-200"
      : "border-zinc-300/45 bg-zinc-200/12 text-zinc-100";
  const pickerTargetLabel =
    rivalPickerTarget === "p1" ? "Jugador P1" : isSpectate ? "Jugador P2" : "Rival";
  const resultPerspective = useMemo<"victory" | "defeat" | "draw" | "spectate">(() => {
    if (outcome === 0 || outcome === null) {
      return "draw";
    }
    if (isSpectate) {
      return "spectate";
    }
    return outcome === PLAYER_1 ? "victory" : "defeat";
  }, [isSpectate, outcome]);
  const resultTitle =
    resultPerspective === "victory"
      ? "VICTORIA"
      : resultPerspective === "defeat"
        ? "DERROTA"
        : resultPerspective === "draw"
          ? "EMPATE"
          : "RESULTADO";
  const resultToneClass =
    resultPerspective === "victory"
      ? "text-lime-200 drop-shadow-[0_0_22px_rgba(163,230,53,0.58)]"
      : resultPerspective === "defeat"
        ? "text-red-200 drop-shadow-[0_0_22px_rgba(248,113,113,0.54)]"
        : "text-zinc-200 drop-shadow-[0_0_18px_rgba(255,255,255,0.2)]";
  const resultPanelToneClass =
    resultPerspective === "victory"
      ? "border-lime-300/45 bg-lime-400/10"
      : resultPerspective === "defeat"
        ? "border-red-400/45 bg-red-500/10"
        : "border-zinc-400/35 bg-zinc-200/10";
  const isEliteWin =
    resultPerspective === "victory" &&
    queueRanked &&
    finishRatingDelta !== null &&
    finishRatingDelta >= 18;
  const isResultOverlayOpen = gameFinished && !showIntro;
  const showDeferredResultStats = !queueRanked || !loadingFinishRewards;

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
  const filteredPlayers = useMemo(() => {
    const query = rivalPickerQuery.trim().toLowerCase();
    if (query.length === 0) {
      return publicPlayers;
    }
    return publicPlayers.filter((player) => player.username.toLowerCase().includes(query));
  }, [rivalPickerQuery, publicPlayers]);
  const selectablePlayers = useMemo(() => {
    if (rivalPickerTarget !== "p2" || isSpectate || user?.id === undefined) {
      return filteredPlayers;
    }
    return filteredPlayers.filter((player) => player.user_id !== user.id);
  }, [filteredPlayers, isSpectate, rivalPickerTarget, user?.id]);

  useEffect(() => {
    let cancelled = false;
    if (
      !gameFinished ||
      !queueRanked ||
      persistedGameId === null ||
      accessToken === null ||
      user?.id === undefined
    ) {
      setLoadingFinishRewards(false);
      return;
    }
    setLoadingFinishRewards(true);
    void (async () => {
      try {
        const seasonId = ratingBaseline?.seasonId ?? (await fetchActiveSeason()).id;
        const maxAttempts = 20;
        let resolved = false;
        for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
          if (cancelled) {
            return;
          }
          const events = await withRetry(
            () => fetchRatingEvents(user.id, seasonId, 30, 0),
            { attempts: 3, baseDelayMs: 160 },
          );
          const foundEvent = events.items.find((item) => item.game_id === persistedGameId);
          if (foundEvent !== undefined) {
            setFinishRatingDelta(foundEvent.delta);
            if (foundEvent.before_lp !== null && foundEvent.after_lp !== null) {
              setFinishLpDelta(foundEvent.after_lp - foundEvent.before_lp);
            } else {
              setFinishLpDelta(null);
            }
            setFinishTransitionType(foundEvent.transition_type);
            resolved = true;
            break;
          }

          if (ratingBaseline !== null && ratingBaseline.seasonId === seasonId) {
            const currentRating = await withRetry(
              () => fetchUserRating(user.id, seasonId),
              { attempts: 3, baseDelayMs: 160 },
            );
            const ratingDelta = currentRating.rating - ratingBaseline.rating;
            const lpDelta = currentRating.lp - ratingBaseline.lp;
            const ratingChanged = Math.abs(ratingDelta) >= 0.01;
            const lpChanged = lpDelta !== 0;
            const divisionChanged =
              currentRating.league !== ratingBaseline.league ||
              currentRating.division !== ratingBaseline.division;
            const gameCountChanged = currentRating.games_played > ratingBaseline.gamesPlayed;

            if (ratingChanged || lpChanged || divisionChanged || gameCountChanged) {
              setFinishRatingDelta(ratingDelta);
              setFinishLpDelta(lpDelta);
              if (divisionChanged) {
                setFinishTransitionType(ratingDelta >= 0 || lpDelta >= 0 ? "promotion" : "demotion");
              } else {
                setFinishTransitionType("stable");
              }
              resolved = true;
              break;
            }
          }

          if (attempt < maxAttempts) {
            await sleep(350);
          }
        }
        if (cancelled || resolved) {
          return;
        }

        // Deterministic fallback: avoid indefinite "Pendiente" when no rating event arrives.
        if (ratingBaseline !== null) {
          try {
            const currentRating = await withRetry(
              () => fetchUserRating(user.id, seasonId),
              { attempts: 3, baseDelayMs: 180 },
            );
            setFinishRatingDelta(currentRating.rating - ratingBaseline.rating);
            setFinishLpDelta(currentRating.lp - ratingBaseline.lp);
            const divisionChanged =
              currentRating.league !== ratingBaseline.league ||
              currentRating.division !== ratingBaseline.division;
            if (divisionChanged) {
              const ratingDelta = currentRating.rating - ratingBaseline.rating;
              const lpDelta = currentRating.lp - ratingBaseline.lp;
              setFinishTransitionType(ratingDelta >= 0 || lpDelta >= 0 ? "promotion" : "demotion");
            } else {
              setFinishTransitionType("stable");
            }
            return;
          } catch {
            // Last-resort fallback below.
          }
        }

        setFinishRatingDelta(0);
        setFinishLpDelta(0);
        setFinishTransitionType("stable");
      } catch {
        if (!cancelled) {
          if (ratingBaseline !== null) {
            try {
              const seasonId = ratingBaseline.seasonId;
              const currentRating = await withRetry(
                () => fetchUserRating(user.id, seasonId),
                { attempts: 6, baseDelayMs: 240 },
              );
              const ratingDelta = currentRating.rating - ratingBaseline.rating;
              const lpDelta = currentRating.lp - ratingBaseline.lp;
              const divisionChanged =
                currentRating.league !== ratingBaseline.league ||
                currentRating.division !== ratingBaseline.division;
              setFinishRatingDelta(ratingDelta);
              setFinishLpDelta(lpDelta);
              if (divisionChanged) {
                setFinishTransitionType(ratingDelta >= 0 || lpDelta >= 0 ? "promotion" : "demotion");
              } else {
                setFinishTransitionType("stable");
              }
            } catch {
              setFinishRatingDelta(0);
              setFinishLpDelta(0);
              setFinishTransitionType("stable");
            }
          } else {
            setFinishRatingDelta(0);
            setFinishLpDelta(0);
            setFinishTransitionType("stable");
          }
        }
      } finally {
        if (!cancelled) {
          setLoadingFinishRewards(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken, gameFinished, persistedGameId, queueRanked, ratingBaseline, user?.id]);

  useEffect(() => {
    if (!loadingFinishRewards || !queueRanked || !gameFinished) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setLoadingFinishRewards(false);
      setFinishRatingDelta((current) => (current === null ? 0 : current));
      setFinishLpDelta((current) => (current === null ? 0 : current));
      setFinishTransitionType((current) => current ?? "stable");
    }, 15000);
    return () => window.clearTimeout(timeoutId);
  }, [gameFinished, loadingFinishRewards, queueRanked]);

  useEffect(() => {
    if (!isResultOverlayOpen) {
      resultRevealPlayedRef.current = false;
      return;
    }
    if (resultRevealPlayedRef.current) {
      return;
    }
    resultRevealPlayedRef.current = true;
    playSfx(SFX.resultReveal, resultPerspective === "defeat" ? 0.24 : 0.2);
  }, [isResultOverlayOpen, resultPerspective]);

  useEffect(() => {
    const animateNumber = (
      target: number | null,
      setter: (value: number | null) => void,
      durationMs: number,
    ): (() => void) | undefined => {
      if (!isResultOverlayOpen || target === null) {
        setter(target);
        return;
      }
      const start = 0;
      const startedAt = performance.now();
      let frameId = 0;
      const tick = (now: number): void => {
        const progress = Math.min(1, (now - startedAt) / durationMs);
        const eased = 1 - (1 - progress) ** 3;
        setter(start + (target - start) * eased);
        if (progress < 1) {
          frameId = window.requestAnimationFrame(tick);
        } else {
          setter(target);
        }
      };
      frameId = window.requestAnimationFrame(tick);
      return () => window.cancelAnimationFrame(frameId);
    };

    const cleanupLp = animateNumber(finishLpDelta, setAnimatedLpDelta, 780);
    const cleanupRating = animateNumber(finishRatingDelta, setAnimatedRatingDelta, 920);
    return () => {
      cleanupLp?.();
      cleanupRating?.();
    };
  }, [finishLpDelta, finishRatingDelta, isResultOverlayOpen]);

  useEffect(() => {
    if (!rivalPickerOpen) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setRivalPickerOpen(false);
      }
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [rivalPickerOpen]);

  const persistSnapshot = useCallback(
    (nextGameId: string | null, nextPending: PendingPersistOperation[]) => {
      if (nextGameId === null && nextPending.length === 0) {
        localStorage.removeItem(PERSIST_SNAPSHOT_KEY);
        return;
      }
      localStorage.setItem(
        PERSIST_SNAPSHOT_KEY,
        JSON.stringify({
          gameId: nextGameId,
          pending: nextPending,
        }),
      );
    },
    [],
  );

  const clearPendingQueue = useCallback(
    (nextGameId: string | null) => {
      failedPersistOpsRef.current = [];
      setPendingPersistCount(0);
      persistSnapshot(nextGameId, []);
    },
    [persistSnapshot],
  );

  const pushPendingOperation = useCallback(
    (operation: PendingPersistOperation, nextGameId: string | null) => {
      failedPersistOpsRef.current.push(operation);
      const nextPending = [...failedPersistOpsRef.current];
      setPendingPersistCount(nextPending.length);
      persistSnapshot(nextGameId, nextPending);
    },
    [persistSnapshot],
  );

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(MATCHMAKING_PRESET_KEY);
      if (raw === null) {
        return;
      }
      sessionStorage.removeItem(MATCHMAKING_PRESET_KEY);
      const preset = JSON.parse(raw) as MatchmakingPreset;
      setMatchMode("play");
      if (preset.opponentProfile === "heuristic" || preset.opponentProfile === "model") {
        setOpponentProfile(preset.opponentProfile);
      }
      if (preset.mode === "fast" || preset.mode === "strong") {
        setMode(preset.mode);
      }
      if (
        preset.heuristicLevel === "easy" ||
        preset.heuristicLevel === "normal" ||
        preset.heuristicLevel === "hard"
      ) {
        setHeuristicLevel(preset.heuristicLevel);
      }
      if (preset.guestMode) {
        setGuestQueueMode(true);
        setQueueRanked(false);
        setGuestRivalName(
          typeof preset.guestRivalName === "string" && preset.guestRivalName.trim().length > 0
            ? preset.guestRivalName.trim()
            : null,
        );
        if (typeof preset.guestUsername === "string" && preset.guestUsername.trim().length > 0) {
          setGuestUsername(preset.guestUsername.trim());
        }
        setQueueNotice("Rival casual encontrado. Preparando tablero...");
      } else {
        setQueueRanked(true);
        setQueueNotice("Rival encontrado. Preparando tablero...");
      }
      setAutoQueueStart(true);
    } catch {
      sessionStorage.removeItem(MATCHMAKING_PRESET_KEY);
    }
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(PERSIST_SNAPSHOT_KEY);
      if (raw === null) {
        return;
      }
      const parsed = JSON.parse(raw) as {
        gameId?: unknown;
        pending?: unknown;
      };
      const restoredGameId = typeof parsed.gameId === "string" && parsed.gameId.length > 0 ? parsed.gameId : null;
      const restoredPending = Array.isArray(parsed.pending)
        ? (parsed.pending.filter(
            (item): item is PendingPersistOperation =>
              typeof item === "object" &&
              item !== null &&
              typeof (item as { gameId?: unknown }).gameId === "string" &&
              typeof (item as { actorLabel?: unknown }).actorLabel === "string" &&
              typeof (item as { beforeBoard?: unknown }).beforeBoard === "object" &&
              typeof (item as { move?: unknown }).move === "object" &&
              typeof (item as { mode?: unknown }).mode === "string",
          ) as PendingPersistOperation[])
        : [];
      if (restoredGameId !== null) {
        setPersistedGameId(restoredGameId);
      }
      if (restoredPending.length > 0) {
        failedPersistOpsRef.current = restoredPending;
        setPendingPersistCount(restoredPending.length);
        setPersistStatus(`Hay ${restoredPending.length} jugadas pendientes por sincronizar.`);
      }
    } catch {
      localStorage.removeItem(PERSIST_SNAPSHOT_KEY);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    if (!isAuthenticated || accessToken === null) {
      if (mounted) {
        setBotAccounts([]);
        setSelectedP1BotId("");
        setSelectedP2BotId("");
      }
      return () => {
        mounted = false;
      };
    }
    void (async () => {
      try {
        const players = await fetchPublicPlayers(accessToken, { limit: 200 });
        if (!mounted) {
          return;
        }
        setPublicPlayers(players);
        const bots: PlayableBot[] = players
          .filter((player) => player.is_bot && player.agent_type !== null)
          .map((player) => ({
            user_id: player.user_id,
            username: player.username,
            bot_kind: player.bot_kind,
            agent_type: player.agent_type ?? "heuristic",
            heuristic_level: player.heuristic_level,
            model_mode: player.model_mode,
            enabled: player.enabled ?? true,
          }));
        setBotAccounts(bots);
        if (bots.length > 0) {
          setSelectedP1BotId((prev) => (prev.length > 0 ? prev : bots[0].user_id));
          setSelectedP2BotId((prev) => (prev.length > 0 ? prev : bots[0].user_id));
        }
      } catch {
        if (mounted) {
          setPublicPlayers([]);
          setBotAccounts([]);
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, [accessToken, isAuthenticated]);

  useEffect(() => {
    if (isSpectate || user?.id === undefined || selectedP2BotId !== user.id) {
      return;
    }
    const firstNonSelf = publicPlayers.find((player) => player.user_id !== user.id);
    setSelectedP2BotId(firstNonSelf?.user_id ?? "");
  }, [isSpectate, publicPlayers, selectedP2BotId, user?.id]);

  useEffect(() => {
    if (!isAuthenticated || accessToken === null || queuedGameId === null) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const game = await fetchPersistedGameSummary(accessToken, queuedGameId);
        if (cancelled) {
          return;
        }
        const isLocalP1 = user?.id !== undefined && game.player1_id === user.id;
        const isLocalP2 = user?.id !== undefined && game.player2_id === user.id;

        if (isLocalP1) {
          setLocalPlayerSide(PLAYER_1);
        } else if (isLocalP2) {
          setLocalPlayerSide(PLAYER_2);
        } else {
          setLocalPlayerSide(PLAYER_1);
        }

        setQueuePlayerAgents({
          p1: game.player1_agent,
          p2: game.player2_agent,
        });

        const opponentId =
          isLocalP1 ? game.player2_id : isLocalP2 ? game.player1_id : game.player2_id;
        if (opponentId !== null) {
          setSelectedP2BotId(opponentId);
        }

        const opponentAgent =
          isLocalP1 ? game.player2_agent : isLocalP2 ? game.player1_agent : game.player2_agent;
        if (opponentAgent === "model" || opponentAgent === "heuristic") {
          setOpponentProfile(opponentAgent);
        }
      } catch {
        // Best effort sync from persisted game metadata.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken, isAuthenticated, queuedGameId, user?.id]);

  useEffect(() => {
    if (!isAuthenticated || accessToken === null || outgoingInvitation === null) {
      return;
    }
    let cancelled = false;
    const intervalId = window.setInterval(() => {
      void (async () => {
        try {
          const invitation = await fetchInvitationGame(accessToken, outgoingInvitation.gameId);
          if (cancelled) {
            return;
          }
          if (invitation.status === "in_progress") {
            setOutgoingInvitation(null);
            setQueuedGameId(invitation.id);
            setQueueRanked(Boolean(invitation.rated));
            setQueueNotice(`${outgoingInvitation.opponentUsername} acepto la invitacion. Entrando a la arena...`);
            setAutoQueueStart(true);
            emitFlash("Invitacion aceptada. Iniciando partida 1v1.", "success");
            return;
          }
          if (invitation.status === "aborted") {
            setOutgoingInvitation(null);
            setQueueNotice(`${outgoingInvitation.opponentUsername} rechazo la invitacion.`);
            emitFlash("La invitacion fue rechazada.", "warning");
            return;
          }
          setStatus(`Solicitud enviada a ${outgoingInvitation.opponentUsername}. Espera su respuesta...`);
        } catch {
          // Keep polling; temporary network errors should not close invitation flow.
        }
      })();
    }, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [accessToken, emitFlash, isAuthenticated, outgoingInvitation]);

  useEffect(() => {
    if (!isQueueMatchView || !isAuthenticated || accessToken === null || user?.id === undefined) {
      setQueueArenaP1Elo(null);
      setQueueArenaP2Elo(null);
      setQueueArenaLoading(false);
      return;
    }
    if (selectedP2Player === null) {
      setQueueArenaP1Elo(null);
      setQueueArenaP2Elo(null);
      setQueueArenaLoading(false);
      return;
    }
    let cancelled = false;
    setQueueArenaLoading(true);
    void (async () => {
      try {
        const season = await fetchActiveSeason();
        const [p1Rating, p2Rating] = await Promise.all([
          fetchUserRating(user.id, season.id),
          fetchUserRating(selectedP2Player.user_id, season.id),
        ]);
        if (cancelled) {
          return;
        }
        setQueueArenaP1Elo(p1Rating.rating);
        setQueueArenaP2Elo(p2Rating.rating);
      } catch {
        if (cancelled) {
          return;
        }
        setQueueArenaP1Elo(null);
        setQueueArenaP2Elo(null);
      } finally {
        if (!cancelled) {
          setQueueArenaLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken, isAuthenticated, isQueueMatchView, selectedP2Player, user?.id]);

  useEffect(() => {
    if (selectedP1Bot === null) {
      return;
    }
    if (selectedP1Bot.agent_type === "model") {
      setP1Profile("model");
      if (selectedP1Bot.model_mode === "fast" || selectedP1Bot.model_mode === "strong") {
        setP1ModelMode(selectedP1Bot.model_mode);
      }
      return;
    }
    setP1Profile("heuristic");
    if (
      selectedP1Bot.heuristic_level === "easy" ||
      selectedP1Bot.heuristic_level === "normal" ||
      selectedP1Bot.heuristic_level === "hard"
    ) {
      setP1HeuristicLevel(selectedP1Bot.heuristic_level);
    }
  }, [selectedP1Bot]);

  useEffect(() => {
    if (selectedP2Bot === null) {
      return;
    }
    if (selectedP2Bot.agent_type === "model") {
      setOpponentProfile("model");
      if (selectedP2Bot.model_mode === "fast" || selectedP2Bot.model_mode === "strong") {
        setMode(selectedP2Bot.model_mode);
      }
      return;
    }
    setOpponentProfile("heuristic");
    if (
      selectedP2Bot.heuristic_level === "easy" ||
      selectedP2Bot.heuristic_level === "normal" ||
      selectedP2Bot.heuristic_level === "hard"
    ) {
      setHeuristicLevel(selectedP2Bot.heuristic_level);
    }
  }, [selectedP2Bot]);

  useEffect(() => {
    const interval = window.setInterval(() => setNowMs(Date.now()), 40);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!matchStarted || !showIntro) {
      return;
    }
    const timeout = window.setTimeout(() => {
      if (introCountdown > 0) {
        setIntroCountdown((prev) => prev - 1);
        return;
      }
      setShowIntro(false);
      setStatus(matchMode === "spectate" ? "Comienza la simulacion IA vs IA!" : "Comienza la partida. Tu turno, comandante.");
    }, introCountdown === 0 ? 520 : 760);

    return () => window.clearTimeout(timeout);
  }, [introCountdown, matchMode, matchStarted, showIntro]);

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
    setStatus("Partida reiniciada. Configura y pulsa Iniciar.");
    setPersistedGameId(null);
    setPersistStatus("Modo local (sin persistencia remota).");
    setPersistError(null);
    setRetryingPersist(false);
    clearPendingQueue(null);
    setReplay(null);
    setPreviewMove(null);
    setPreviewUntil(0);
    setInfectionMask({});
    setInfectionBursts([]);
    setLastResolvedMove(null);
    setMatchStarted(false);
    setShowIntro(false);
    setIntroCountdown(INTRO_COUNTDOWN_START);
    setMatchStartMs(null);
    setMatchEndMs(null);
    setResolvedMoves(0);
    setQueueNotice(null);
    setOutgoingInvitation(null);
    setAutoQueueStart(false);
    setQueueRanked(false);
    setQueuedGameId(null);
    setQueuePlayerAgents(null);
    setLocalPlayerSide(PLAYER_1);
    setGuestQueueMode(false);
    setGuestRivalName(null);
    setQueueRedirecting(false);
    setRatingBaseline(null);
    setFinishLpDelta(null);
    setFinishRatingDelta(null);
    setFinishTransitionType(null);
    setLoadingFinishRewards(false);
    setAnimatedLpDelta(null);
    setAnimatedRatingDelta(null);
  }, [clearPendingQueue]);

  const consumeQueuedMatchFromSession = useCallback(async (): Promise<void> => {
    try {
      const rawMatch = sessionStorage.getItem(MATCHMAKING_MATCH_KEY);
      if (rawMatch === null) {
        return;
      }
      sessionStorage.removeItem(MATCHMAKING_MATCH_KEY);
      const payload = JSON.parse(rawMatch) as MatchmakingMatch;
      if (typeof payload.gameId !== "string" || payload.gameId.length === 0) {
        return;
      }

      if (matchStarted || persistedGameId !== null) {
        if (accessToken !== null && persistedGameId !== null) {
          try {
            await deletePersistedGame(accessToken, persistedGameId);
          } catch {
            // Best effort remote close before switching to accepted invitation game.
          }
        }
        resetGame();
      }

      setQueuedGameId(payload.gameId);
      setQueueRanked(true);
      setQueueNotice(
        payload.matchedWith === "human"
          ? "Rival en cola encontrado. Preparando tablero..."
          : "Rival bot encontrado. Preparando tablero...",
      );
      setAutoQueueStart(true);
    } catch {
      sessionStorage.removeItem(MATCHMAKING_MATCH_KEY);
    }
  }, [accessToken, matchStarted, persistedGameId, resetGame]);

  useEffect(() => {
    void consumeQueuedMatchFromSession();
  }, [consumeQueuedMatchFromSession, location.key]);

  useEffect(() => {
    if (!isAuthenticated || accessToken === null || persistedGameId === null) {
      gameplayWsRef.current?.close();
      gameplayWsRef.current = null;
      lastWsPlyRef.current = -1;
      return;
    }

    const onEvent = (event: PersistedGameWsEvent): void => {
      if (event.type === "game.closed") {
        if (event.game_id !== persistedGameId || exitingMatch) {
          return;
        }
        resetGame();
        navigate("/", {
          state: {
            flash: {
              message: "El rival salio de la partida. Volviste al lobby.",
              tone: "warning",
            },
          },
        });
        return;
      }
      if (event.type !== "game.move.applied") {
        return;
      }
      if (event.game_id !== persistedGameId) {
        return;
      }
      if (event.move.ply <= lastWsPlyRef.current) {
        return;
      }
      lastWsPlyRef.current = event.move.ply;

      if (event.move.board_after !== null) {
        const boardAfter = event.move.board_after as BoardState;
        setBoard(boardAfter);
      }
      const remoteMove =
        event.move.r1 === null || event.move.c1 === null || event.move.r2 === null || event.move.c2 === null
          ? null
          : { r1: event.move.r1, c1: event.move.c1, r2: event.move.r2, c2: event.move.c2 };
      if (remoteMove !== null) {
        setPreviewMove(remoteMove);
        setPreviewUntil(Date.now() + AI_PREVIEW_MS);
      }
      setResolvedMoves((prev) => Math.max(prev, event.move.ply + 1));
      setLastResolvedMove(remoteMove);

      if (event.game.status === "finished") {
        setMatchEndMs(Date.now());
        setStatus("Partida finalizada.");
      } else {
        setStatus(`Sincronizado en vivo: jugada ${event.move.ply + 1}.`);
      }
    };

    const socket = openPersistedGameSocket(accessToken, persistedGameId, onEvent);
    gameplayWsRef.current = socket;

    socket.onclose = () => {
      if (gameplayWsRef.current === socket) {
        gameplayWsRef.current = null;
      }
    };

    return () => {
      if (gameplayWsRef.current === socket) {
        gameplayWsRef.current = null;
      }
      socket.close();
    };
  }, [accessToken, exitingMatch, isAuthenticated, navigate, persistedGameId, resetGame]);

  const leaveMatch = useCallback(
    async (options?: { redirectTo?: string; logoutAfter?: boolean }) => {
      if (exitingMatch) {
        return;
      }
      setExitingMatch(true);
      try {
        if (accessToken !== null && persistedGameId !== null) {
          try {
            await deletePersistedGame(accessToken, persistedGameId);
          } catch (error) {
            const message =
              error instanceof Error
                ? error.message
                : "No se pudo cerrar remotamente la partida. Saliendo en local.";
            setPersistError(message);
            emitFlash("No se pudo cerrar la partida remota. Se cerrara en local.", "warning");
          }
        }
        resetGame();
        setShowExitConfirm(false);
        setPendingNavigationPath(null);
        setPendingLogout(false);
        if (options?.logoutAfter) {
          await logout();
          navigate("/auth/login", {
            replace: true,
            state: { flash: { message: "Sesion cerrada.", tone: "info" } },
          });
          return;
        }
        if (options?.redirectTo) {
          navigate(options.redirectTo, {
            state: { flash: { message: "Partida cerrada.", tone: "info" } },
          });
          return;
        }
        navigate("/", {
          state: { flash: { message: "Partida cerrada.", tone: "info" } },
        });
      } finally {
        setExitingMatch(false);
      }
    },
    [accessToken, emitFlash, exitingMatch, logout, navigate, persistedGameId, resetGame],
  );

  const requestExitForNavigation = useCallback(
    (path: string): boolean => {
      if (!hasBlockingRemoteMatch) {
        return true;
      }
      if (path === location.pathname) {
        return true;
      }
      setPendingNavigationPath(path);
      setPendingLogout(false);
      setShowExitConfirm(true);
      return false;
    },
    [hasBlockingRemoteMatch, location.pathname],
  );

  const requestExitForLogout = useCallback((): boolean => {
    if (!hasBlockingRemoteMatch) {
      return true;
    }
    setPendingNavigationPath(null);
    setPendingLogout(true);
    setShowExitConfirm(true);
    return false;
  }, [hasBlockingRemoteMatch]);

  useEffect(() => {
    if (!hasBlockingRemoteMatch) {
      return;
    }
    const onBeforeUnload = (event: BeforeUnloadEvent): void => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [hasBlockingRemoteMatch]);

  useEffect(() => {
    return () => {
      if (unmountCleanupTriggeredRef.current) {
        return;
      }
      const snapshot = unmountCleanupStateRef.current;
      const tokenToUse = snapshot.accessToken ?? lastAccessTokenRef.current;
      if (
        tokenToUse === null ||
        snapshot.persistedGameId === null ||
        !snapshot.matchStarted ||
        snapshot.gameFinished ||
        snapshot.exitingMatch
      ) {
        return;
      }
      unmountCleanupTriggeredRef.current = true;
      void deletePersistedGame(tokenToUse, snapshot.persistedGameId);
    };
  }, []);

  const randomizeSpectatorBots = useCallback(() => {
    if (botAccounts.length === 0) {
      return;
    }
    if (botAccounts.length === 1) {
      setSelectedP1BotId(botAccounts[0].user_id);
      setSelectedP2BotId(botAccounts[0].user_id);
      return;
    }
    const firstIdx = Math.floor(Math.random() * botAccounts.length);
    let secondIdx = Math.floor(Math.random() * botAccounts.length);
    if (secondIdx === firstIdx) {
      secondIdx = (secondIdx + 1) % botAccounts.length;
    }
    setSelectedP1BotId(botAccounts[firstIdx].user_id);
    setSelectedP2BotId(botAccounts[secondIdx].user_id);
  }, [botAccounts]);

  const swapSpectatorBots = useCallback(() => {
    if (selectedP1BotId.length === 0 && selectedP2BotId.length === 0) {
      return;
    }
    setSelectedP1BotId(selectedP2BotId);
    setSelectedP2BotId(selectedP1BotId);
  }, [selectedP1BotId, selectedP2BotId]);

  const activateAutoPersistence = useCallback(async () => {
    if (!isAuthenticated || accessToken === null) {
      setPersistStatus("Modo local (inicia sesion para guardar partidas automaticamente).");
      return;
    }
    setPersisting(true);
    setPersistStatus("Activando guardado remoto...");
    try {
      if (isSpectate && (selectedP1Bot === null || selectedP2Bot === null)) {
        setPersistStatus("Selecciona cuentas IA para P1 y P2 antes de iniciar espectador.");
        return;
      }
      if (queuedGameId !== null) {
        setPersistedGameId(queuedGameId);
        clearPendingQueue(queuedGameId);
        setPersistStatus(`Partida ranked enlazada: ${queuedGameId.slice(0, 8)}...`);
        setPersistError(null);
        return;
      }
      const gameId = await createPersistedGame(accessToken, opponentProfile, {
        ranked: queueRanked,
        preferredHeuristicLevel: heuristicLevel,
        preferredModelMode: mode,
        selectedP1BotUserId: isSpectate ? selectedP1Bot?.user_id : undefined,
        selectedP2BotUserId: selectedP2Bot?.user_id,
        player1Agent: isSpectate ? p1Profile : "human",
        player2Agent: opponentProfile,
      });
      setPersistedGameId(gameId);
      clearPendingQueue(gameId);
      setPersistStatus(`Guardado remoto activo: ${gameId.slice(0, 8)}...`);
      setPersistError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "No se pudo activar guardado remoto";
      setPersistStatus(`Guardado remoto deshabilitado: ${message}`);
    } finally {
      setPersisting(false);
    }
  }, [
    accessToken,
    clearPendingQueue,
    heuristicLevel,
    isAuthenticated,
    isSpectate,
    mode,
    opponentProfile,
    p1Profile,
    queueRanked,
    queuedGameId,
    selectedP1Bot,
    selectedP2Bot,
  ]);

  const startMatch = useCallback(async (): Promise<boolean> => {
    const selectedRivalIsHuman = selectedP2Player !== null && !selectedP2Player.is_bot;
    if (queuedGameId === null && selectedRivalIsHuman) {
      if (!isAuthenticated || accessToken === null || selectedP2Player === null) {
        setStatus("Inicia sesion para enviar una solicitud 1v1.");
        emitFlash("Debes iniciar sesion para invitar jugadores humanos.", "warning");
        return false;
      }
      try {
        const invitation = await createHumanInvitation(accessToken, selectedP2Player.user_id);
        setOutgoingInvitation({
          gameId: invitation.id,
          opponentUserId: selectedP2Player.user_id,
          opponentUsername: selectedP2Player.username,
          rated: queueRanked,
        });
        setQueueNotice(`Invitacion enviada a ${selectedP2Player.username}.`);
        setStatus(`Solicitud enviada a ${selectedP2Player.username}. Espera su respuesta...`);
        emitFlash(`Solicitud 1v1 enviada a ${selectedP2Player.username}.`, "success");
      } catch (error: unknown) {
        const message =
          error instanceof Error ? error.message : "No se pudo enviar la solicitud.";
        setStatus(message);
        emitFlash(message, "error");
      }
      return false;
    }
    if (isSpectate && selectedP1Bot === null) {
      setStatus("Modo observador requiere que P1 y P2 sean bots.");
      emitFlash("Para IA vs IA debes seleccionar dos cuentas bot.", "warning");
      return false;
    }
    playSfx(SFX.start, 0.32);
    setQueueNotice(null);
    setOutgoingInvitation(null);
    setRatingBaseline(null);
    setFinishLpDelta(null);
    setFinishRatingDelta(null);
    setFinishTransitionType(null);
    setLoadingFinishRewards(false);
    setBoard(createInitialBoard());
    setSelected(null);
    setEvalValue(null);
    setPersistedGameId(null);
    setPersistStatus("Preparando sesion...");
    setPersistError(null);
    setRetryingPersist(false);
    clearPendingQueue(null);
    setReplay(null);
    setPreviewMove(null);
    setPreviewUntil(0);
    setInfectionMask({});
    setInfectionBursts([]);
    setLastResolvedMove(null);
    setIntroCountdown(INTRO_COUNTDOWN_START);
    setShowIntro(true);
    setMatchStarted(true);
    setStatus("Preparando partida...");
    setMatchStartMs(Date.now());
    setMatchEndMs(null);
    setResolvedMoves(0);
    if (queueRanked && isAuthenticated && accessToken !== null && user?.id !== undefined) {
      try {
        const season = await fetchActiveSeason();
        const currentRating = await fetchUserRating(user.id, season.id);
        setRatingBaseline({
          seasonId: season.id,
          rating: currentRating.rating,
          lp: currentRating.lp,
          league: currentRating.league,
          division: currentRating.division,
          gamesPlayed: currentRating.games_played,
        });
      } catch {
        setRatingBaseline(null);
      }
    }
    await activateAutoPersistence();
    return true;
  }, [
    accessToken,
    activateAutoPersistence,
    clearPendingQueue,
    emitFlash,
    isAuthenticated,
    isSpectate,
    queuedGameId,
    queueRanked,
    selectedP1Bot,
    selectedP2Player,
    user?.id,
  ]);

  const backToQueue = useCallback(() => {
    const shouldAutoQueue = queueRanked && isAuthenticated;
    playSfx(SFX.queueDeploy, 0.26);
    setQueueRedirecting(true);
    void (async () => {
      await sleep(420);
      resetGame();
      if (shouldAutoQueue) {
        sessionStorage.setItem(
          MATCHMAKING_AUTOQUEUE_KEY,
          JSON.stringify({ autoJoin: true, createdAt: Date.now() }),
        );
        navigate("/", {
          state: { flash: { message: "Reingresando a cola competitiva...", tone: "info" } },
        });
        return;
      }
      navigate("/", {
        state: { flash: { message: "Volviste al lobby.", tone: "info" } },
      });
    })();
  }, [isAuthenticated, navigate, queueRanked, resetGame]);

  useEffect(() => {
    if (!autoQueueStart || matchStarted || thinking || persisting) {
      return;
    }
    void (async () => {
      const started = await startMatch();
      if (started) {
        setAutoQueueStart(false);
      }
    })();
  }, [
    autoQueueStart,
    matchStarted,
    persisting,
    startMatch,
    thinking,
  ]);

  const applyBoardUpdate = useCallback(
    (next: BoardState, passCount: number) => {
      setBoard(next);
      setSelected(null);
      if (isGameOver(next)) {
        const result = getOutcome(next);
        if (isSpectate) {
          if (result === 0) {
            playSfx(SFX.uiClick, 0.2);
          } else {
            playSfx(SFX.win, 0.35);
          }
        } else if (result === PLAYER_1) {
          playSfx(SFX.win, 0.35);
        } else if (result === PLAYER_2) {
          playSfx(SFX.lose, 0.35);
        } else {
          playSfx(SFX.uiClick, 0.2);
        }
        setMatchEndMs(Date.now());
        setStatus(`${winnerLabel(getOutcome(next))}. Fin de partida.`);
        if (persistedGameId !== null) {
          setPersistStatus("Partida finalizada. Sincronizando replay...");
        }
        return;
      }
      if (passCount > 0) {
        setStatus(`Pase automatico (${passCount}) por falta de jugadas legales.`);
        return;
      }
      const controller = sideController(next.current_player);
      if (controller === "human") {
        setStatus("Turno humano: define tu movimiento.");
      } else if (controller === "remote_human") {
        setStatus("Esperando jugada del rival...");
      } else {
        setStatus(`Turno IA (${controller}): analizando vector...`);
      }
    },
    [isSpectate, persistedGameId, sideController],
  );

  const animateTransition = useCallback(
    (before: BoardState, next: BoardState, move: Move | null, actingPlayer: 1 | -1) => {
      const mask = buildInfectionMask(before, next, move, actingPlayer, Date.now());
      if (move !== null && Object.keys(mask).length > 0) {
        playSfx(SFX.infect, 0.16);
      } else if (move !== null && actingPlayer === PLAYER_2) {
        const now = Date.now();
        if (now - lastP2MoveSfxAtRef.current >= P2_MOVE_SFX_MIN_GAP_MS) {
          playSfx(SFX.moveLand, 0.18);
          lastP2MoveSfxAtRef.current = now;
        }
      }
      setInfectionMask(mask);
      setLastResolvedMove(move);
      if (move !== null) {
        setResolvedMoves((prev) => prev + 1);
      }
      const normalized = normalizeForcedPasses(next);
      applyBoardUpdate(normalized.board, normalized.passes);
    },
    [applyBoardUpdate],
  );

  const persistMoveWithRetry = useCallback(
    async (operation: PendingPersistOperation) => {
      if (!canPersist || accessToken === null) {
        throw new Error("Persistencia no disponible.");
      }
      let lastError: unknown = null;
      for (let attempt = 1; attempt <= PERSIST_MAX_RETRIES; attempt += 1) {
        try {
          await storeManualMove(
            accessToken,
            operation.gameId,
            operation.beforeBoard,
            operation.move,
            operation.mode,
          );
          setPersistError(null);
          return;
        } catch (error) {
          lastError = error;
          if (isFatalPersistenceError(error)) {
            break;
          }
          if (attempt < PERSIST_MAX_RETRIES) {
            const backoffMs = PERSIST_BASE_BACKOFF_MS * 2 ** (attempt - 1);
            await sleep(backoffMs);
          }
        }
      }
      throw lastError;
    },
    [accessToken, canPersist],
  );

  const disableRemotePersistence = useCallback(
    (message: string) => {
      setPersistedGameId(null);
      setPersistStatus("Persistencia remota desactivada. Continuas en modo local.");
      setPersistError(message);
      clearPendingQueue(null);
      emitFlash("La partida remota ya no existe. Continuas en modo local.", "warning");
    },
    [clearPendingQueue, emitFlash],
  );

  const enqueuePersistManualMove = useCallback(
    (beforeBoard: BoardState, move: Move, mode: PersistedMoveMode, actorLabel: string) => {
      if (!canPersist || accessToken === null || persistedGameId === null) {
        return;
      }
      const operation: PendingPersistOperation = {
        gameId: persistedGameId,
        beforeBoard,
        move,
        mode,
        actorLabel,
      };
      persistQueueRef.current = persistQueueRef.current
        .then(async () => {
          await persistMoveWithRetry(operation);
        })
        .catch((error) => {
          const message =
            error instanceof Error ? error.message : "Error desconocido al guardar la jugada";
          if (isFatalPersistenceError(error)) {
            disableRemotePersistence(`Persistencia detenida: ${message}`);
            return;
          }
          setPersistError(`No se pudo guardar ${actorLabel}: ${message}`);
          setPersistStatus("Persistencia con fallos: revisa conexion/API.");
          pushPendingOperation(operation, persistedGameId);
        });
    },
    [
      accessToken,
      canPersist,
      disableRemotePersistence,
      persistedGameId,
      persistMoveWithRetry,
      pushPendingOperation,
    ],
  );

  const commitHumanMove = useCallback(
    async (candidate: Move) => {
      setThinking(true);
      const before = board;
      try {
        const nextBoard = applyMove(board, candidate);
        animateTransition(before, nextBoard, candidate, board.current_player as 1 | -1);
        enqueuePersistManualMove(before, candidate, "manual", "la jugada del equipo humano");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Invalid move";
        setStatus(message);
      } finally {
        setThinking(false);
      }
    },
    [animateTransition, board, enqueuePersistManualMove],
  );

  const runAIMove = useCallback(async () => {
    setThinking(true);
    const before = board;
    const side = board.current_player as 1 | -1;
    const controller = sideController(side);
    try {
      if (controller === "human" || controller === "remote_human") {
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

      const nextBoard = applyMove(board, plannedMove);
      if (plannedMove !== null) {
        let actorLabel = "jugada IA";
        if (controller === "model") {
          actorLabel = side === PLAYER_1 ? "la jugada del modelo (P1)" : "la jugada del modelo (P2)";
        } else if (controller === "heuristic") {
          actorLabel =
            side === PLAYER_1
              ? "la jugada heuristica (P1)"
              : "la jugada heuristica (P2)";
        }
        enqueuePersistManualMove(before, plannedMove, requestMode, actorLabel);
      }

      setPreviewMove(null);
      setPreviewUntil(0);
      animateTransition(before, nextBoard, plannedMove, side);
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
    enqueuePersistManualMove,
    sideController,
    sideMoveMode,
  ]);

  useEffect(() => {
    if (!matchStarted || interactionLocked || isGameOver(board) || showIntro) {
      return;
    }
    const turnController = sideController(board.current_player);
    if (turnController === "human" || turnController === "remote_human") {
      return;
    }
    void runAIMove();
  }, [board, interactionLocked, matchStarted, runAIMove, showIntro, sideController]);

  const refreshReplay = useCallback(async () => {
    if (!canPersist || accessToken === null || persistedGameId === null) {
      return;
    }
    setThinking(true);
    try {
      const payload = await fetchPersistedReplay(accessToken, persistedGameId);
      setReplay(payload);
      setPersistStatus(`Replay sincronizado. Jugadas: ${payload.moves.length}`);
      setPersistError(null);
      emitFlash("Replay sincronizada.", "success");
    } catch (error) {
      const message = error instanceof Error ? error.message : "No se pudo obtener el replay";
      setPersistStatus(`Replay error: ${message}`);
      emitFlash("No se pudo sincronizar la replay.", "error");
    } finally {
      setThinking(false);
    }
  }, [accessToken, canPersist, emitFlash, persistedGameId]);

  useEffect(() => {
    if (!gameFinished || persistedGameId === null || accessToken === null || !canPersist) {
      return;
    }
    void refreshReplay();
  }, [accessToken, canPersist, gameFinished, persistedGameId, refreshReplay]);

  const retryFailedPersistence = useCallback(async () => {
    if (failedPersistOpsRef.current.length === 0 || persistedGameId === null) {
      return;
    }
    setRetryingPersist(true);
    const pendingForGame = failedPersistOpsRef.current.filter(
      (operation) => operation.gameId === persistedGameId,
    );
    if (pendingForGame.length === 0) {
      setRetryingPersist(false);
      setPersistStatus("No hay pendientes para esta partida.");
      return;
    }
    setPersistStatus(`Reintentando ${pendingForGame.length} jugadas pendientes...`);
    const pending = [...pendingForGame];
    failedPersistOpsRef.current = [];
    setPendingPersistCount(0);
    persistSnapshot(persistedGameId, []);
    for (const operation of pending) {
      try {
        await persistMoveWithRetry(operation);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Error desconocido al reintentar sincronizacion";
        if (isFatalPersistenceError(error)) {
          disableRemotePersistence(`Persistencia detenida: ${message}`);
          setRetryingPersist(false);
          return;
        }
        failedPersistOpsRef.current.push(operation);
        setPersistError(`Sigue pendiente ${operation.actorLabel}: ${message}`);
        setPersistStatus("Aun hay jugadas sin sincronizar.");
      }
    }
    setPendingPersistCount(failedPersistOpsRef.current.length);
    persistSnapshot(persistedGameId, [...failedPersistOpsRef.current]);
    if (failedPersistOpsRef.current.length === 0) {
      setPersistError(null);
      setPersistStatus("Sincronizacion completada.");
      if (gameFinished) {
        void refreshReplay();
      }
    }
    setRetryingPersist(false);
  }, [
    disableRemotePersistence,
    gameFinished,
    persistMoveWithRetry,
    persistedGameId,
    persistSnapshot,
    refreshReplay,
  ]);

  const onCellClick = useCallback(
    (row: number, col: number) => {
      if (
        !matchStarted ||
        isSpectate ||
        interactionLocked ||
        isGameOver(board) ||
        sideController(board.current_player) !== "human" ||
        showIntro
      ) {
        return;
      }
      const cell = board.grid[row][col];
      if (cell === board.current_player) {
        playSfx(SFX.uiClick, 0.24);
        setSelected([row, col]);
        return;
      }
      if (selected === null) {
        return;
      }
      const candidate = selectedMoves.find((move) => move.r2 === row && move.c2 === col);
      if (!candidate) {
        if (selected[0] === row && selected[1] === col) {
          playSfx(SFX.uiClick, 0.2);
          setSelected(null);
        }
        return;
      }
      playSfx(SFX.uiClick, 0.2);
      void commitHumanMove(candidate);
    },
    [board, commitHumanMove, interactionLocked, isSpectate, matchStarted, selected, selectedMoves, showIntro, sideController],
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

  const onStartButtonHover = useCallback(() => {
    if (matchStarted) {
      return;
    }
    const now = Date.now();
    if (now - lastHoverSfxAtRef.current < HOVER_SFX_MIN_GAP_MS) {
      return;
    }
    lastHoverSfxAtRef.current = now;
    playSfx(SFX.uiHover, 1.0);
  }, [matchStarted]);

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
    <AppShell onNavigateAttempt={requestExitForNavigation} onLogoutAttempt={requestExitForLogout}>
      <AnimatePresence>
        {showExitConfirm ? (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 px-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16, ease: "easeOut" }}
            onClick={() => {
              if (!exitingMatch) {
                setShowExitConfirm(false);
              }
            }}
          >
            <motion.div
              className="w-full max-w-sm rounded-xl border border-zinc-700/80 bg-zinc-950/95 p-4 shadow-[0_0_30px_rgba(0,0,0,0.55)]"
              initial={{ opacity: 0, y: 8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 6, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              onClick={(event) => event.stopPropagation()}
            >
              <p className="text-xs uppercase tracking-[0.14em] text-zinc-400">Salir de partida</p>
              <h3 className="mt-1 text-base font-semibold text-zinc-100">Confirmar salida</h3>
              <p className="mt-2 text-sm text-zinc-300">
                {pendingLogout
                  ? "Se cerrara la partida en curso y luego tu sesion."
                  : pendingNavigationPath
                    ? "Se cerrara la partida en curso y cambiaras de pestaña."
                    : "Se cerrara la partida en curso y volveras al inicio."}
              </p>
              <div className="mt-4 flex items-center justify-end gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    setShowExitConfirm(false);
                    setPendingNavigationPath(null);
                    setPendingLogout(false);
                  }}
                  disabled={exitingMatch}
                >
                  Cancelar
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="border border-red-500/55 bg-red-500/15 text-red-200 hover:bg-red-500/25"
                  onClick={() => {
                    void leaveMatch({
                      redirectTo: pendingNavigationPath ?? undefined,
                      logoutAfter: pendingLogout,
                    });
                  }}
                  disabled={exitingMatch}
                >
                  {exitingMatch ? "Saliendo..." : "Si, salir"}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <div className={`grid gap-4 ${isQueueMatchView ? "grid-cols-1" : "lg:grid-cols-[minmax(0,1fr)_340px]"}`}>
        <Card className="overflow-hidden border-zinc-800/90 bg-gradient-to-b from-zinc-950/95 to-black/95">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <div>
                <Badge className="mb-2 border-lime-400/35 bg-lime-300/10 text-lime-200" variant="default">
                  UNDERBYTELABS // ATAXX-ZERO
                </Badge>
                <CardTitle>Arena Ataxx</CardTitle>
                <CardDescription>
                  {isSpectate
                    ? `Simulacion IA vs IA (P1 ${p1Profile} vs P2 ${opponentProfile}).`
                    : isHumanVsHumanMatch
                      ? "Humano vs humano (1v1 invitacion)."
                      : `Humano vs IA (${opponentProfile}).`}
                </CardDescription>
                {queueNotice ? <p className="mt-1 text-sm text-primary">{queueNotice}</p> : null}
              </div>
              <Button variant="secondary" size="sm" onClick={resetGame}>
                Reset
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {isQueueMatchView ? (
              <div className="mx-auto mb-3 grid w-full max-w-[620px] grid-cols-3 items-center gap-2 rounded-xl border border-lime-300/35 bg-gradient-to-r from-lime-400/10 via-lime-300/5 to-lime-400/10 p-2 shadow-[0_0_18px_rgba(163,230,53,0.16)]">
                <div className="rounded-md border border-zinc-700/80 bg-zinc-950/90 px-2 py-1.5">
                  <p className="truncate text-[11px] uppercase tracking-[0.1em] text-zinc-500">Jugador</p>
                  <p className="truncate text-sm font-semibold text-zinc-100" title={`@${playerDisplayName}`}>
                    @{playerDisplayName}
                  </p>
                  <p className="text-xs font-medium text-lime-300">
                    ELO {queueArenaLoading ? "..." : queueArenaP1Elo !== null ? queueArenaP1Elo.toFixed(1) : "-"}
                  </p>
                </div>
                <div className="text-center text-xs font-semibold uppercase tracking-[0.22em] text-lime-200 drop-shadow-[0_0_8px_rgba(163,230,53,0.45)]">
                  vs
                </div>
                <div className="rounded-md border border-zinc-700/80 bg-zinc-950/90 px-2 py-1.5">
                  <p className="truncate text-[11px] uppercase tracking-[0.1em] text-zinc-500">Rival</p>
                  <p className="truncate text-sm font-semibold text-zinc-100" title={selectedP2Player?.username ?? guestRivalName ?? "rival"}>
                    {selectedP2Player?.username ?? guestRivalName ?? "rival"}
                  </p>
                  <p className="text-xs font-medium text-lime-300">
                    ELO {queueArenaLoading ? "..." : queueArenaP2Elo !== null ? queueArenaP2Elo.toFixed(1) : "-"}
                  </p>
                </div>
              </div>
            ) : null}
            <motion.div
              className={`relative aspect-square max-w-[620px] [perspective:1000px] ${isQueueMatchView ? "mx-auto" : ""}`}
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
                    <p className="mb-2 text-xs uppercase tracking-[0.22em] text-zinc-400">Iniciando partida</p>
                    <motion.p
                      key={`intro-${introCountdown}`}
                      className="text-6xl font-black text-lime-200 drop-shadow-[0_0_18px_rgba(132,204,22,0.65)]"
                      initial={{ scale: 0.72, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 1.2, opacity: 0 }}
                      transition={{ duration: 0.3, ease: "easeOut" }}
                    >
                      {introCountdown > 0 ? introCountdown : "LUCHA!"}
                    </motion.p>
                  </motion.div>
                )}
                {gameFinished && !showIntro && (
                  <motion.div
                    className="absolute inset-0 z-40 flex items-center justify-center bg-black/55 px-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <motion.div
                      className={`pointer-events-auto relative w-full max-w-lg overflow-hidden rounded-2xl border bg-zinc-950/95 px-5 py-5 text-left shadow-[0_0_40px_rgba(132,204,22,0.25)] ${resultPanelToneClass}`}
                      initial={{ y: 10, scale: 0.96, opacity: 0 }}
                      animate={{ y: 0, scale: 1, opacity: 1 }}
                      transition={{ duration: 0.28, ease: "easeOut" }}
                    >
                      <motion.div
                        className="pointer-events-none absolute -left-10 -top-16 h-36 w-36 rounded-full bg-lime-300/10 blur-2xl"
                        animate={{ scale: [0.92, 1.06, 0.92], opacity: [0.35, 0.6, 0.35] }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                      />
                      <motion.div
                        className="pointer-events-none absolute -right-8 -bottom-14 h-32 w-32 rounded-full bg-zinc-100/10 blur-2xl"
                        animate={{ scale: [1.04, 0.94, 1.04], opacity: [0.28, 0.5, 0.28] }}
                        transition={{ duration: 3.4, repeat: Infinity, ease: "easeInOut" }}
                      />

                      <div className="relative">
                        <motion.p
                          className="text-center text-[11px] uppercase tracking-[0.28em] text-zinc-400"
                          animate={{ opacity: [0.7, 1, 0.7] }}
                          transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
                        >
                          Resultado Final
                        </motion.p>
                        <motion.div
                          className="pointer-events-none absolute left-1/2 top-0 h-14 w-14 -translate-x-1/2 rounded-full border border-lime-300/30"
                          initial={{ scale: 0.35, opacity: 0 }}
                          animate={{ scale: [0.5, 1.05, 1.32], opacity: [0.38, 0.2, 0] }}
                          transition={{ duration: 1.1, ease: "easeOut" }}
                        />
                        <motion.div
                          className="pointer-events-none absolute left-1/2 top-0 h-20 w-20 -translate-x-1/2 rounded-full border border-zinc-200/20"
                          initial={{ scale: 0.4, opacity: 0 }}
                          animate={{ scale: [0.5, 1.1, 1.45], opacity: [0.2, 0.12, 0] }}
                          transition={{ duration: 1.4, ease: "easeOut", delay: 0.08 }}
                        />
                        <motion.p
                          className={`mt-2 text-center text-4xl font-black tracking-[0.08em] ${resultToneClass}`}
                          initial={{ scale: 0.9, opacity: 0, x: 0 }}
                          animate={{
                            scale:
                              resultPerspective === "victory"
                                ? [0.92, 1.06, 1]
                                : [1, 1.02, 1],
                            opacity: 1,
                            x: resultPerspective === "defeat" ? [0, -1, 1, 0] : [0, 0],
                            letterSpacing:
                              resultPerspective === "victory" ? ["0.28em", "0.12em", "0.08em"] : ["0.08em", "0.08em"],
                            filter:
                              resultPerspective === "victory"
                                ? ["drop-shadow(0 0 0px rgba(163,230,53,0))", "drop-shadow(0 0 18px rgba(163,230,53,0.6))", "drop-shadow(0 0 10px rgba(163,230,53,0.35))"]
                                : ["drop-shadow(0 0 0px rgba(248,113,113,0))", "drop-shadow(0 0 12px rgba(248,113,113,0.38))", "drop-shadow(0 0 7px rgba(248,113,113,0.22))"],
                          }}
                          transition={{ duration: 0.6, ease: "easeOut" }}
                        >
                          {resultTitle}
                        </motion.p>
                        {resultPerspective === "defeat" && (
                          <motion.div
                            className="pointer-events-none absolute inset-x-0 top-10 h-16"
                            style={{
                              background:
                                "repeating-linear-gradient(180deg, rgba(248,113,113,0.16) 0px, rgba(248,113,113,0.16) 1px, rgba(0,0,0,0) 2px, rgba(0,0,0,0) 4px)",
                            }}
                            animate={{ opacity: [0.08, 0.22, 0.08, 0], y: [0, 2, 0, 0] }}
                            transition={{ duration: 0.9, ease: "easeOut" }}
                          />
                        )}
                        {resultPerspective === "victory" && (
                          <motion.p
                            className="mt-1 text-center text-xs uppercase tracking-[0.2em] text-lime-300/90"
                            initial={{ opacity: 0, y: 3 }}
                            animate={{ opacity: [0, 1, 0.78], y: 0 }}
                            transition={{ duration: 0.65, ease: "easeOut", delay: 0.05 }}
                          >
                            recompensa confirmada
                          </motion.p>
                        )}
                        {isEliteWin && (
                          <motion.p
                            className="mt-1 text-center text-xs font-semibold uppercase tracking-[0.18em] text-lime-300"
                            initial={{ opacity: 0, y: 3 }}
                            animate={{ opacity: [0.5, 1, 0.5], y: 0 }}
                            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                          >
                            Kernel Break
                          </motion.p>
                        )}

                        <p className="mt-2 text-center text-base font-semibold text-zinc-100">
                          {winnerName === null ? (
                            winnerLabel(outcome)
                          ) : (
                            <>
                              Gana{" "}
                              <span
                                className={`inline-flex items-center rounded-md border px-2 py-0.5 text-base font-bold ${winnerHighlightClass}`}
                              >
                                {winnerName}
                              </span>
                            </>
                          )}
                        </p>

                        {!showDeferredResultStats ? (
                          <motion.div
                            className="relative mt-4 overflow-hidden rounded-xl border border-lime-300/25 bg-zinc-900/65 px-3 py-3"
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.28, ease: "easeOut" }}
                          >
                            <div className="relative flex items-center justify-between gap-3">
                              <p className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-lime-200/95">
                                <motion.span
                                  className="h-1.5 w-1.5 rounded-full bg-lime-300"
                                  animate={{ opacity: [0.35, 1, 0.35] }}
                                  transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                                />
                                Sincronizando recompensa
                              </p>
                              <Activity className="h-4 w-4 text-lime-300/75" />
                            </div>
                            <p className="relative mt-1 text-xs text-zinc-400">
                              Confirmando LP y cerrando reporte final.
                            </p>
                            <motion.div
                              className="relative mt-3 h-px bg-zinc-800"
                              animate={{ opacity: [0.45, 0.75, 0.45] }}
                              transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                            >
                              <motion.div
                                className="absolute left-0 top-0 h-px bg-lime-300/85"
                                initial={{ width: "20%" }}
                                animate={{ width: ["20%", "62%", "34%"] }}
                                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                              />
                            </motion.div>
                          </motion.div>
                        ) : (
                          <>
                            {queueRanked && (
                              <motion.div
                                className={`mt-3 rounded-xl border px-3 py-3 ${
                                  finishLpDelta === null
                                    ? "border-zinc-600/70 bg-zinc-900/65"
                                    : finishLpDelta >= 0
                                      ? "border-lime-400/55 bg-lime-500/10"
                                      : "border-red-400/50 bg-red-500/10"
                                }`}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.32, ease: "easeOut", delay: 0.06 }}
                              >
                                <p className="text-[11px] uppercase tracking-[0.2em] text-zinc-400">Recompensa Competitiva</p>
                                <div className="mt-1 flex items-center justify-between">
                                  <p className="inline-flex items-center gap-1 text-sm text-zinc-200">
                                    <Crown className="h-4 w-4 text-lime-300" />
                                    LP
                                  </p>
                                  <p
                                    className={`text-lg font-bold ${
                                      animatedLpDelta === null
                                        ? "text-zinc-300"
                                        : Math.round(animatedLpDelta) === 0
                                          ? "text-zinc-200"
                                          : animatedLpDelta > 0
                                          ? "text-lime-200"
                                          : "text-red-200"
                                    }`}
                                  >
                                    {animatedLpDelta === null
                                      ? "Pendiente"
                                      : `${formatSignedDelta(Math.round(animatedLpDelta))} LP`}
                                  </p>
                                </div>
                                {animatedLpDelta !== null &&
                                  Math.round(animatedLpDelta) === 0 &&
                                  animatedRatingDelta !== null &&
                                  Math.abs(animatedRatingDelta) >= 0.05 && (
                                    <p className="mt-1 text-[11px] text-zinc-400">
                                      Sin cambio de LP (ajuste parcial de MMR).
                                    </p>
                                  )}
                                {animatedRatingDelta !== null && (
                                  <p className="mt-1 text-xs text-zinc-300">
                                    MMR: <span className="font-semibold">{formatSignedDelta(animatedRatingDelta)}</span>
                                  </p>
                                )}
                                {finishTransitionType !== null && (
                                  <p className="mt-1 text-xs text-zinc-400">
                                    Estado:{" "}
                                    <span className="font-semibold text-zinc-200">
                                      {finishTransitionType === "promotion"
                                        ? "Ascenso"
                                        : finishTransitionType === "demotion"
                                          ? "Descenso"
                                          : "Estable"}
                                    </span>
                                  </p>
                                )}
                              </motion.div>
                            )}

                            <div className="mt-4 grid grid-cols-2 gap-2">
                              <div className="rounded-lg border border-zinc-700/90 bg-zinc-900/75 p-3">
                                <p className="text-[11px] uppercase tracking-wider text-zinc-400">{p1SummaryLabel}</p>
                                <p className="mt-1 text-xl font-bold text-zinc-100">{counts.p1}</p>
                                <p className="text-[11px] text-zinc-400">puntos</p>
                              </div>
                              <div className="rounded-lg border border-lime-400/40 bg-lime-500/10 p-3">
                                <p className="text-[11px] uppercase tracking-wider text-lime-200/90">{p2SummaryLabel}</p>
                                <p className="mt-1 text-xl font-bold text-lime-300">{counts.p2}</p>
                                <p className="text-[11px] text-lime-200/70">puntos</p>
                              </div>
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-2">
                              <div className="rounded-lg border border-zinc-700/80 bg-zinc-900/65 px-3 py-2">
                                <p className="text-[11px] uppercase tracking-wider text-zinc-400">Duracion</p>
                                <p className="mt-1 text-sm font-semibold text-zinc-100">{matchDurationSeconds}s</p>
                              </div>
                              <div className="rounded-lg border border-zinc-700/80 bg-zinc-900/65 px-3 py-2">
                                <p className="text-[11px] uppercase tracking-wider text-zinc-400">Jugadas</p>
                                <p className="mt-1 text-sm font-semibold text-zinc-100">{resolvedMoves}</p>
                              </div>
                            </div>

                            <p className="mt-3 rounded-lg border border-zinc-700/80 bg-zinc-900/70 px-3 py-2 text-sm text-zinc-200">
                              {winnerName === null ? (
                                "La partida termino en empate."
                              ) : (
                                <>
                                  Ganador confirmado:{" "}
                                  <span className="font-semibold text-lime-200">{winnerName}</span>.
                                </>
                              )}
                            </p>
                          </>
                        )}
                      </div>

                      {isQueueMatchView ? (
                        <div className="mt-4 grid grid-cols-1 gap-2">
                          <Button
                            type="button"
                            className="border border-lime-300/70 bg-lime-300 text-black hover:bg-lime-200"
                            disabled={queueRedirecting}
                            onClick={() => {
                              playSfx(SFX.queueDeploy, 0.26);
                              setQueueRedirecting(true);
                              void (async () => {
                                await sleep(380);
                                resetGame();
                                sessionStorage.setItem(
                                  MATCHMAKING_AUTOQUEUE_KEY,
                                  JSON.stringify({ autoJoin: true, createdAt: Date.now() }),
                                );
                                navigate("/", {
                                  state: { flash: { message: "Reingresando a cola competitiva...", tone: "info" } },
                                });
                              })();
                            }}
                          >
                            <span className="inline-flex items-center gap-1.5">
                              <ArrowRight className="h-4 w-4" />
                              {queueRedirecting ? "Volviendo..." : "Volver a buscar partida"}
                            </span>
                          </Button>
                        </div>
                      ) : (
                        <div className="mt-4 grid grid-cols-3 gap-2">
                          <Button
                            type="button"
                            className="border border-lime-300/70 bg-lime-300 text-black hover:bg-lime-200"
                            disabled={queueRedirecting}
                            onClick={backToQueue}
                          >
                            <span className="inline-flex items-center gap-1.5">
                              <ArrowRight className="h-4 w-4" />
                              {queueRedirecting ? "Conectando..." : "Volver a cola"}
                            </span>
                          </Button>
                          <Button
                            type="button"
                            className="border border-zinc-400/70 bg-zinc-200 text-black hover:bg-zinc-100"
                            disabled={queueRedirecting}
                            onClick={() => {
                              playSfx(SFX.uiClick, 0.24);
                              resetGame();
                            }}
                          >
                            Lobby
                          </Button>
                          <Button
                            type="button"
                            className="border border-lime-200/70 bg-lime-300 text-black hover:bg-lime-200"
                            disabled={queueRedirecting}
                            onClick={() => {
                              playSfx(SFX.uiClick, 0.24);
                              void startMatch();
                            }}
                          >
                            <span className="inline-flex items-center gap-1.5">
                              <RotateCcw className="h-4 w-4" />
                              Revancha
                            </span>
                          </Button>
                        </div>
                      )}
                      <AnimatePresence>
                        {queueRedirecting ? (
                          <motion.div
                            className="pointer-events-none absolute inset-0 rounded-2xl bg-black/70"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                          >
                            <div className="flex h-full items-center justify-center">
                              <motion.div
                                className="h-12 w-12 rounded-full border border-lime-300/55 border-t-lime-100"
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                              />
                            </div>
                          </motion.div>
                        ) : null}
                      </AnimatePresence>
                    </motion.div>
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
                  const canPick =
                    matchStarted &&
                    !isSpectate &&
                    sideController(board.current_player) === "human" &&
                    cell === board.current_player;
                  const isPreviewOrigin = previewMove !== null && previewMove.r1 === r && previewMove.c1 === c;
                  const isPreviewTarget = previewMove !== null && previewMove.r2 === r && previewMove.c2 === c;
                  const isRecentOrigin = lastOrigin !== null && lastOrigin.row === r && lastOrigin.col === c;
                  const isRecentTarget = lastTarget !== null && lastTarget.row === r && lastTarget.col === c;

                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onCellClick(r, c)}
                      disabled={!matchStarted || isSpectate || interactionLocked || isGameOver(board)}
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

        {!isQueueMatchView ? (
          <Card className="border-zinc-800/90 bg-black/90">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Gauge className="h-4 w-4 text-lime-300" />
              Panel de Partida
            </CardTitle>
            <CardDescription>Estado, ritmo y control en tiempo real.</CardDescription>
          </CardHeader>
          <CardContent>
            <motion.div
              className="space-y-4"
              initial="hidden"
              animate="show"
              variants={panelSectionVariants}
              custom={0}
            >
            <motion.div
              className="rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-950/95 to-black/95 p-2.5 hover:border-zinc-700"
              variants={panelSectionVariants}
              custom={0.03}
            >
              <div className="grid grid-cols-2 gap-2">
                <div className="min-h-[74px] rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Modo</p>
                  <p className="mt-1 text-sm font-semibold text-zinc-100">{modeLabel}</p>
                </div>
                <div className="min-h-[74px] rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Cola</p>
                  <p className="mt-1 text-sm font-semibold text-lime-200">{matchTypeLabel}</p>
                </div>
                <div className="min-h-[74px] rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                    {isSpectate ? "Controlador P1" : "Jugador"}
                  </p>
                  <p className="mt-1 truncate text-sm font-semibold text-zinc-100" title={playerLabel}>
                    {playerLabel}
                  </p>
                </div>
                <div className="min-h-[74px] rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                    {isSpectate ? "Controlador P2" : "Rival"}
                  </p>
                  <p className="mt-1 truncate text-sm font-semibold text-zinc-100" title={rivalDisplayLabel}>
                    {rivalDisplayLabel}
                  </p>
                </div>
              </div>
            </motion.div>

            <motion.div className="grid grid-cols-2 gap-2 text-xs" variants={panelSectionVariants} custom={0.06}>
              <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                <p className="inline-flex items-center gap-1 text-zinc-400">
                  <Swords className="h-3.5 w-3.5 text-lime-300" />
                  Turno
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-100">{currentTurnLabel}</p>
              </div>
              <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                <p className="inline-flex items-center gap-1 text-zinc-400">
                  <Activity className="h-3.5 w-3.5 text-lime-300" />
                  Estado
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-100">{winnerLabel(outcome)}</p>
              </div>
              <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                <p className="inline-flex items-center gap-1 text-zinc-400">
                  <User className="h-3.5 w-3.5 text-zinc-200" />
                  Fichas P1
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-100">{counts.p1}</p>
              </div>
              <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                <p className="inline-flex items-center gap-1 text-zinc-400">
                  <Bot className="h-3.5 w-3.5 text-lime-300" />
                  Fichas P2
                </p>
                <p className="mt-1 text-sm font-semibold text-lime-300">{counts.p2}</p>
              </div>
              <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                <p className="inline-flex items-center gap-1 text-zinc-400">
                  <Timer className="h-3.5 w-3.5 text-lime-300" />
                  Ritmo
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-100">{matchPace}</p>
              </div>
              <div className="rounded-md border border-zinc-700/80 bg-zinc-950/80 p-2">
                <p className="inline-flex items-center gap-1 text-zinc-400">
                  <Brain className="h-3.5 w-3.5 text-lime-300" />
                  Eval IA
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-100">
                  {evalValue === null ? "-" : evalValue.toFixed(3)}
                </p>
              </div>
            </motion.div>

            <motion.div
              className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-2.5 hover:border-zinc-700"
              animate={{ borderColor: ["rgba(63,63,70,0.9)", "rgba(132,204,22,0.35)", "rgba(63,63,70,0.9)"] }}
              transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
              variants={panelSectionVariants}
              custom={0.09}
            >
              <div className="mb-1 flex items-center justify-between text-xs text-zinc-400">
                <span className="inline-flex items-center gap-1">
                  <ShieldAlert className="h-3.5 w-3.5 text-lime-300" />
                  Control enemigo
                </span>
                <span>
                  {threatLevel}% | delta {controlDeltaLabel}
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-zinc-900">
                <motion.div
                  className="h-full bg-gradient-to-r from-lime-500 to-lime-300"
                  initial={false}
                  animate={{ width: `${threatLevel}%` }}
                  transition={{ duration: 0.35, ease: "easeOut" }}
                />
              </div>
              <p className="mt-1 text-[11px] text-zinc-500">
                Medios turnos: {board.half_moves} | Jugadas resueltas: {resolvedMoves}
              </p>
            </motion.div>

            <motion.div className="space-y-2" variants={panelSectionVariants} custom={0.12}>
              <div>
                <label htmlFor="match-mode" className="mb-1 block text-xs text-zinc-400">
                  Modo de partida
                </label>
                <select
                  id="match-mode"
                  value={matchMode}
                  onChange={(event) => {
                    const nextMode = event.target.value as MatchMode;
                    setMatchMode(nextMode);
                    if (nextMode === "spectate") {
                      setQueueRanked(false);
                    }
                  }}
                  className="h-9 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100 transition hover:border-zinc-500 focus:border-lime-400/60 focus:outline-none focus:ring-1 focus:ring-lime-400/40"
                  disabled={interactionLocked || matchStarted}
                >
                  <option value="play">Jugar (tu vs jugador seleccionado)</option>
                  <option value="spectate">Observador (jugador vs jugador)</option>
                </select>
              </div>
              <div>
                <label htmlFor="match-queue-toggle" className="mb-1 block text-xs text-zinc-400">
                  Tipo de partida
                </label>
                <div
                  id="match-queue-toggle"
                  className={`relative grid h-9 grid-cols-2 rounded-md border bg-zinc-950 p-0.5 ${
                    isSpectate ? "border-zinc-800" : "border-lime-400/30"
                  }`}
                  aria-label="Tipo de partida"
                >
                  <motion.div
                    className="absolute bottom-0.5 left-0.5 top-0.5 rounded-md bg-lime-300/90"
                    initial={false}
                    animate={{ x: queueRanked && !isSpectate ? "100%" : "0%" }}
                    transition={{ type: "spring", stiffness: 260, damping: 24 }}
                    style={{ width: "calc(50% - 2px)" }}
                  />
                  <button
                    type="button"
                    onClick={() => setQueueRanked(false)}
                    disabled={interactionLocked || matchStarted || isSpectate}
                    className={`relative z-10 rounded text-sm font-semibold transition ${
                      !queueRanked ? "text-black" : "text-zinc-300 hover:text-zinc-100"
                    } ${isSpectate ? "cursor-not-allowed opacity-50" : ""}`}
                  >
                    Casual
                  </button>
                  <button
                    type="button"
                    onClick={() => setQueueRanked(true)}
                    disabled={interactionLocked || matchStarted || isSpectate}
                    className={`relative z-10 rounded text-sm font-semibold transition ${
                      queueRanked ? "text-black" : "text-zinc-300 hover:text-zinc-100"
                    } ${isSpectate ? "cursor-not-allowed opacity-50" : ""}`}
                  >
                    Ranked
                  </button>
                </div>
                {!isSpectate ? (
                  <motion.p
                    className="mt-1 text-[11px] text-zinc-500"
                    animate={{ opacity: [0.55, 1, 0.55] }}
                    transition={{ duration: 1.9, repeat: Infinity, ease: "easeInOut" }}
                  >
                    Toca el modo para alternar rapido.
                  </motion.p>
                ) : null}
                {isSpectate ? (
                  <p className="mt-1 text-[11px] text-zinc-500">Observador se guarda como casual para evitar abuso de LP.</p>
                ) : null}
              </div>
              <Button
                type="button"
                onClick={() => {
                  void startMatch();
                }}
                onMouseEnter={onStartButtonHover}
                disabled={interactionLocked || showIntro}
                variant="default"
                size="sm"
                className={`w-full font-semibold tracking-wide transition ${
                  matchStarted
                    ? "opacity-90"
                    : "border border-lime-200/70 bg-lime-300 text-black shadow-[0_0_20px_rgba(163,230,53,0.45)] hover:bg-lime-200"
                }`}
              >
                {matchStarted ? "Partida en curso" : "Iniciar partida"}
              </Button>
              {matchStarted && !gameFinished ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  className="w-full border border-red-500/45 bg-red-500/10 text-red-100 hover:bg-red-500/18"
                  onClick={() => setShowExitConfirm(true)}
                  disabled={interactionLocked || exitingMatch}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Salir de partida
                </Button>
              ) : null}
              {isSpectate ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label htmlFor="p1-opponent" className="mb-1 block text-xs text-zinc-400">
                        Jugador P1
                      </label>
                      <select
                        id="p1-opponent"
                        value={selectedP1BotId}
                        onChange={(event) => setSelectedP1BotId(event.target.value)}
                        className="sr-only"
                        disabled={interactionLocked || matchStarted}
                      >
                        {publicPlayers.length === 0 ? (
                          <option value="">sin jugadores</option>
                        ) : (
                          publicPlayers.map((player) => (
                            <option key={`p1-${player.user_id}`} value={player.user_id}>
                              {playerOptionLabel(player)}
                            </option>
                          ))
                        )}
                      </select>
                      <button
                        type="button"
                        onClick={() => {
                          setRivalPickerTarget("p1");
                          setRivalPickerQuery("");
                          setRivalPickerOpen(true);
                        }}
                        disabled={interactionLocked || matchStarted}
                        className="group flex h-9 w-full items-center justify-between rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100 transition hover:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        <span className="inline-flex items-center gap-2 truncate">
                          <Search className="h-4 w-4 text-zinc-500 transition group-hover:text-lime-300" />
                          <span className="truncate">{selectedP1Player?.username ?? "Seleccionar jugador P1..."}</span>
                        </span>
                        <span className="text-[10px] uppercase tracking-[0.1em] text-zinc-500">
                          {selectedP1Player?.is_bot ? "bot" : "humano"}
                        </span>
                      </button>
                    </div>
                    <div className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-300">
                      <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Perfil P1</p>
                      <p className="mt-1 text-zinc-100">
                        {selectedP1Player === null
                          ? "-"
                          : selectedP1Player.is_bot
                            ? selectedP1Bot?.agent_type === "model"
                              ? `bot modelo | ${selectedP1Bot.model_mode ?? "fast"}`
                              : `bot heuristica | ${selectedP1Bot?.heuristic_level ?? "normal"}`
                            : "humano"}
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      className="w-full"
                      onClick={swapSpectatorBots}
                      disabled={interactionLocked || matchStarted || botAccounts.length === 0}
                    >
                      Swap P1/P2
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      className="w-full"
                      onClick={randomizeSpectatorBots}
                      disabled={interactionLocked || matchStarted || botAccounts.length === 0}
                    >
                      Aleatorio
                    </Button>
                  </div>
                </div>
              ) : null}
            </motion.div>

            <motion.div className="grid grid-cols-2 gap-2" variants={panelSectionVariants} custom={0.15}>
              <div className="space-y-2 rounded-md border border-zinc-800 bg-zinc-950/65 p-2 hover:border-zinc-700">
                <label htmlFor="opponent" className="mb-1 inline-flex items-center gap-1 text-xs text-zinc-400">
                  <User className="h-3.5 w-3.5 text-lime-300/90" />
                  {isSpectate ? "Jugador P2" : "Rival (jugador)"}
                </label>
                <select
                  id="opponent"
                  value={selectedP2BotId}
                  onChange={(event) => setSelectedP2BotId(event.target.value)}
                  className="sr-only"
                  disabled={interactionLocked || matchStarted}
                >
                  {publicPlayers.length === 0 ? (
                    <option value="">sin jugadores</option>
                  ) : (
                    publicPlayers.map((player) => (
                      <option key={`p2-${player.user_id}`} value={player.user_id}>
                        {playerOptionLabel(player)}
                      </option>
                    ))
                  )}
                </select>
                <button
                  type="button"
                  onClick={() => {
                    setRivalPickerTarget("p2");
                    setRivalPickerQuery("");
                    setRivalPickerOpen(true);
                  }}
                  disabled={interactionLocked || matchStarted}
                  className="group flex h-9 w-full items-center justify-between rounded-md border border-zinc-700 bg-zinc-950 px-2 text-sm text-zinc-100 transition hover:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <span className="inline-flex items-center gap-2 truncate">
                    <Search className="h-4 w-4 text-zinc-500 transition group-hover:text-lime-300" />
                    <span className="truncate">{selectedP2Player?.username ?? "Seleccionar rival..."}</span>
                  </span>
                  <span className="text-[10px] uppercase tracking-[0.1em] text-zinc-500">
                    {selectedP2Player?.is_bot ? "bot" : "humano"}
                  </span>
                </button>
              </div>
              <div className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-300 hover:border-zinc-500">
                <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Perfil P2</p>
                <p className="mt-1 truncate text-zinc-100" title={selectedP2Player?.username ?? "-"}>
                  {selectedP2Player?.username ?? "-"}
                </p>
                <p className="mt-1 text-zinc-400">
                  {selectedP2Player === null
                    ? "-"
                    : selectedP2Player.is_bot
                      ? selectedP2Bot?.agent_type === "model"
                        ? `bot modelo | ${selectedP2Bot.model_mode ?? "fast"}`
                        : `bot heuristica | ${selectedP2Bot?.heuristic_level ?? "normal"}`
                      : "humano"}
                </p>
              </div>
            </motion.div>

            <motion.div className="space-y-2" variants={panelSectionVariants} custom={0.18}>
              <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2 hover:border-zinc-500">
                <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Estado de sesion</p>
                <p className="mt-1 text-xs text-zinc-300">
                  {thinking ? `IA activa (${sideController(board.current_player)}) pensando...` : status}
                </p>
              </div>
              <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2 hover:border-zinc-500">
                <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Guardado</p>
                <p className="mt-1 text-[11px] text-zinc-300">
                  {isSpectate ? (
                    "Modo espectador: sin historial remoto."
                  ) : isAuthenticated ? (
                    "Guardado automatico activo al iniciar partida."
                  ) : (
                    <>
                      Modo local activo.{" "}
                      <Link to="/auth/login" className="text-lime-300 underline">
                        Inicia sesion
                      </Link>{" "}
                      para guardar historial.
                    </>
                  )}
                </p>
                {persistedGameId !== null && (
                  <p className="mt-2 break-all text-[11px] text-zinc-400">
                    ID partida: <span className="text-zinc-100">{persistedGameId}</span>
                  </p>
                )}
                <p className="mt-1 text-[11px] text-zinc-500">{persistStatus}</p>
              </div>
            </motion.div>
            {persistError !== null && (
              <div className="rounded-md border border-red-500/50 bg-red-500/10 p-2 text-xs text-red-200">
                {persistError}
                {pendingPersistCount > 0 && (
                  <div className="mt-2 space-y-2">
                    <p className="text-[11px] text-red-100/90">
                      Pendientes: {pendingPersistCount}
                    </p>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      className="w-full"
                      disabled={retryingPersist}
                      onClick={() => {
                        void retryFailedPersistence();
                      }}
                    >
                      {retryingPersist ? "Reintentando..." : "Reintentar sincronizacion"}
                    </Button>
                  </div>
                )}
              </div>
            )}
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
                {persistedGameId !== null && (
                  <Button asChild type="button" size="sm" variant="secondary" className="mt-2 w-full">
                    <Link to={`/profile/games/${persistedGameId}`}>Ver replay detallado</Link>
                  </Button>
                )}
              </div>
            )}
            </motion.div>
          </CardContent>
          </Card>
        ) : null}
      </div>
      <AnimatePresence>
        {rivalPickerOpen ? (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setRivalPickerOpen(false)}
          >
            <motion.div
              className="w-full max-w-xl rounded-2xl border border-zinc-700 bg-zinc-950/98 shadow-[0_20px_70px_rgba(0,0,0,0.55)]"
              initial={{ opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.98 }}
              transition={{ type: "spring", stiffness: 220, damping: 24 }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
                <div>
                  <p className="inline-flex items-center gap-1 text-xs uppercase tracking-[0.12em] text-zinc-500">
                    <Search className="h-3.5 w-3.5 text-lime-300" />
                    Selector de jugador
                  </p>
                  <p className="mt-1 text-sm font-semibold text-zinc-100">
                    Elegir {pickerTargetLabel}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setRivalPickerOpen(false)}
                  className="rounded-md border border-zinc-700 p-1.5 text-zinc-400 transition hover:border-zinc-500 hover:text-zinc-100"
                  aria-label="Cerrar selector de rival"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="p-4">
                <div className="group mb-3 flex h-10 items-center gap-2 rounded-md border border-zinc-700 bg-zinc-950 px-2 transition hover:border-zinc-500 focus-within:border-lime-400/60 focus-within:ring-1 focus-within:ring-lime-400/40">
                  <Search className="h-4 w-4 text-zinc-500 group-focus-within:text-lime-300" />
                  <input
                    type="text"
                    value={rivalPickerQuery}
                    onChange={(event) => setRivalPickerQuery(event.target.value)}
                    placeholder="Buscar por username..."
                    className="h-full w-full bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
                    autoFocus
                  />
                </div>
                <div className="picker-scroll max-h-72 overflow-auto rounded-md border border-zinc-800 bg-zinc-950/80 p-1">
                  {selectablePlayers.length === 0 ? (
                    <p className="px-2 py-3 text-xs text-zinc-500">Sin resultados para la busqueda.</p>
                  ) : (
                    selectablePlayers.map((player) => {
                      const isSelected =
                        rivalPickerTarget === "p1"
                          ? player.user_id === selectedP1BotId
                          : player.user_id === selectedP2BotId;
                      return (
                        <button
                          key={`rival-modal-${player.user_id}`}
                          type="button"
                          onClick={() => {
                            if (rivalPickerTarget === "p1") {
                              setSelectedP1BotId(player.user_id);
                            } else {
                              setSelectedP2BotId(player.user_id);
                            }
                            setRivalPickerOpen(false);
                            setRivalPickerQuery("");
                          }}
                          className={`mb-1 flex w-full items-center justify-between rounded-md border px-2.5 py-2 text-left transition last:mb-0 ${
                            isSelected
                              ? "border-lime-400/55 bg-lime-400/10 text-lime-200"
                              : "border-transparent bg-zinc-950 text-zinc-200 hover:border-zinc-600 hover:bg-zinc-900"
                          }`}
                        >
                          <div className="flex min-w-0 items-center gap-2">
                            {player.is_bot ? (
                              <Bot className="h-3.5 w-3.5 shrink-0 text-lime-300/90" />
                            ) : (
                              <User className="h-3.5 w-3.5 shrink-0 text-zinc-300" />
                            )}
                            <span className="truncate text-sm font-semibold">{player.username}</span>
                          </div>
                          <span className="ml-3 text-[10px] uppercase tracking-[0.12em] text-zinc-500">
                            {player.is_bot ? "bot" : "humano"}
                          </span>
                        </button>
                      );
                    })
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </AppShell>
  );
}




















