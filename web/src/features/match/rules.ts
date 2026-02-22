import type { BoardState, Cell, GameOutcome, Move, Player } from "@/features/match/types";

export const BOARD_SIZE = 7;
export const PLAYER_1: Player = 1;
export const PLAYER_2: Player = -1;
export const EMPTY: Cell = 0;
const HALF_MOVE_CAP = 100;

function cloneGrid(grid: Cell[][]): Cell[][] {
  return grid.map((row) => row.slice()) as Cell[][];
}

export function createInitialBoard(): BoardState {
  const grid: Cell[][] = Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => EMPTY),
  );
  grid[0][0] = PLAYER_1;
  grid[BOARD_SIZE - 1][BOARD_SIZE - 1] = PLAYER_1;
  grid[0][BOARD_SIZE - 1] = PLAYER_2;
  grid[BOARD_SIZE - 1][0] = PLAYER_2;
  return {
    grid,
    current_player: PLAYER_1,
    half_moves: 0,
  };
}

export function opponent(player: Player): Player {
  return (player * -1) as Player;
}

function inBounds(row: number, col: number): boolean {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

function moveDistance(move: Move): number {
  return Math.max(Math.abs(move.r2 - move.r1), Math.abs(move.c2 - move.c1));
}

export function getValidMoves(state: BoardState, player: Player = state.current_player): Move[] {
  const moves: Move[] = [];
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      if (state.grid[r][c] !== player) {
        continue;
      }
      for (let tr = Math.max(0, r - 2); tr <= Math.min(BOARD_SIZE - 1, r + 2); tr += 1) {
        for (let tc = Math.max(0, c - 2); tc <= Math.min(BOARD_SIZE - 1, c + 2); tc += 1) {
          if ((tr !== r || tc !== c) && state.grid[tr][tc] === EMPTY) {
            moves.push({ r1: r, c1: c, r2: tr, c2: tc });
          }
        }
      }
    }
  }
  return moves;
}

export function hasValidMoves(state: BoardState, player: Player = state.current_player): boolean {
  return getValidMoves(state, player).length > 0;
}

function infectNeighbors(grid: Cell[][], row: number, col: number, player: Player): void {
  const enemy = opponent(player);
  for (let r = Math.max(0, row - 1); r <= Math.min(BOARD_SIZE - 1, row + 1); r += 1) {
    for (let c = Math.max(0, col - 1); c <= Math.min(BOARD_SIZE - 1, col + 1); c += 1) {
      if (grid[r][c] === enemy) {
        grid[r][c] = player;
      }
    }
  }
}

export function applyMove(state: BoardState, move: Move | null): BoardState {
  const grid = cloneGrid(state.grid);
  const currentPlayer = state.current_player;

  if (move === null) {
    if (hasValidMoves(state, currentPlayer)) {
      throw new Error("Pass is illegal when legal moves exist.");
    }
    return {
      grid,
      current_player: opponent(currentPlayer),
      half_moves: state.half_moves + 1,
    };
  }

  if (!inBounds(move.r1, move.c1) || !inBounds(move.r2, move.c2)) {
    throw new Error("Move is out of board bounds.");
  }
  if (grid[move.r1][move.c1] !== currentPlayer) {
    throw new Error("Move origin does not belong to current player.");
  }
  if (grid[move.r2][move.c2] !== EMPTY) {
    throw new Error("Move target must be empty.");
  }

  const distance = moveDistance(move);
  if (distance !== 1 && distance !== 2) {
    throw new Error("Illegal move distance.");
  }

  grid[move.r2][move.c2] = currentPlayer;
  if (distance === 2) {
    grid[move.r1][move.c1] = EMPTY;
  }
  infectNeighbors(grid, move.r2, move.c2, currentPlayer);

  return {
    grid,
    current_player: opponent(currentPlayer),
    half_moves: state.half_moves + 1,
  };
}

export function countPieces(state: BoardState): { p1: number; p2: number; empty: number } {
  let p1 = 0;
  let p2 = 0;
  let empty = 0;
  for (const row of state.grid) {
    for (const cell of row) {
      if (cell === PLAYER_1) {
        p1 += 1;
      } else if (cell === PLAYER_2) {
        p2 += 1;
      } else {
        empty += 1;
      }
    }
  }
  return { p1, p2, empty };
}

export function isGameOver(state: BoardState): boolean {
  const counts = countPieces(state);
  if (counts.empty === 0) {
    return true;
  }
  if (counts.p1 === 0 || counts.p2 === 0) {
    return true;
  }
  if (state.half_moves >= HALF_MOVE_CAP) {
    return true;
  }
  return !hasValidMoves(state, state.current_player) && !hasValidMoves(state, opponent(state.current_player));
}

export function getOutcome(state: BoardState): GameOutcome | null {
  if (!isGameOver(state)) {
    return null;
  }
  const counts = countPieces(state);
  if (counts.p1 > counts.p2) {
    return PLAYER_1;
  }
  if (counts.p2 > counts.p1) {
    return PLAYER_2;
  }
  return 0;
}

export function normalizeForcedPasses(state: BoardState): { board: BoardState; passes: number } {
  let current = state;
  let passes = 0;
  while (!isGameOver(current) && !hasValidMoves(current, current.current_player)) {
    current = applyMove(current, null);
    passes += 1;
  }
  return { board: current, passes };
}
