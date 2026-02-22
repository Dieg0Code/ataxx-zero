export type Player = 1 | -1;
export type Cell = Player | 0;

export type BoardState = {
  grid: Cell[][];
  current_player: Player;
  half_moves: number;
};

export type Move = {
  r1: number;
  c1: number;
  r2: number;
  c2: number;
};

export type GameOutcome = 1 | -1 | 0;
