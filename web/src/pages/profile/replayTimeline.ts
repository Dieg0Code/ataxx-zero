import { applyMove, createInitialBoard } from "@/features/match/rules";
import type { PersistedReplay } from "@/features/match/persistence";
import type { BoardState, Move } from "@/features/match/types";

function isBoardState(value: unknown): value is BoardState {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as {
    grid?: unknown;
    current_player?: unknown;
    half_moves?: unknown;
  };
  if (!Array.isArray(candidate.grid) || candidate.grid.length !== 7) {
    return false;
  }
  const rowsValid = candidate.grid.every(
    (row) =>
      Array.isArray(row) &&
      row.length === 7 &&
      row.every((cell) => cell === -1 || cell === 0 || cell === 1),
  );
  if (!rowsValid) {
    return false;
  }
  if (candidate.current_player !== 1 && candidate.current_player !== -1) {
    return false;
  }
  return typeof candidate.half_moves === "number";
}

function replayMoveToMove(move: {
  r1: number | null;
  c1: number | null;
  r2: number | null;
  c2: number | null;
}): Move | null {
  if (move.r1 === null || move.c1 === null || move.r2 === null || move.c2 === null) {
    return null;
  }
  return {
    r1: move.r1,
    c1: move.c1,
    r2: move.r2,
    c2: move.c2,
  };
}

export function getOrderedReplayMoves(replay: PersistedReplay): PersistedReplay["moves"] {
  return [...replay.moves]
    .map((move, index) => ({ move, index }))
    .sort((left, right) => {
      const byPly = left.move.ply - right.move.ply;
      if (byPly !== 0) {
        return byPly;
      }
      return left.index - right.index;
    })
    .map((item) => item.move);
}

export function buildBoardTimeline(replay: PersistedReplay): BoardState[] {
  const orderedMoves = getOrderedReplayMoves(replay);
  if (orderedMoves.length === 0) {
    return [];
  }
  const firstMove = orderedMoves[0];
  const initial =
    (isBoardState(firstMove.board_before) && firstMove.board_before) ||
    (isBoardState(firstMove.board_after) && firstMove.board_after) ||
    createInitialBoard();

  const timeline: BoardState[] = [initial];
  for (const move of orderedMoves) {
    const previous = timeline[timeline.length - 1];
    let baseBoard = previous;
    const expectedPlayer = move.player_side === "p1" ? 1 : -1;
    for (let guard = 0; guard < 2 && baseBoard.current_player !== expectedPlayer; guard += 1) {
      try {
        baseBoard = applyMove(baseBoard, null);
      } catch {
        break;
      }
    }
    try {
      timeline.push(applyMove(baseBoard, replayMoveToMove(move)));
      continue;
    } catch {
      if (isBoardState(move.board_after)) {
        timeline.push(move.board_after);
      } else {
        timeline.push(previous);
      }
    }
  }
  return timeline;
}
