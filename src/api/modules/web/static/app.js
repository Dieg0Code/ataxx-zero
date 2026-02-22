const BOARD_SIZE = 7;
const HUMAN = 1;
const AI = -1;

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const modeEl = document.getElementById("mode");
const resetBtn = document.getElementById("reset");
const aiMoveBtn = document.getElementById("aiMove");

let selected = null;
let thinking = false;
let state = makeInitialState();

function makeInitialState() {
  const grid = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));
  grid[0][0] = HUMAN;
  grid[BOARD_SIZE - 1][BOARD_SIZE - 1] = HUMAN;
  grid[0][BOARD_SIZE - 1] = AI;
  grid[BOARD_SIZE - 1][0] = AI;
  return { grid, current_player: HUMAN, half_moves: 0 };
}

function inBounds(r, c) {
  return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

function chebyshev(r1, c1, r2, c2) {
  return Math.max(Math.abs(r1 - r2), Math.abs(c1 - c2));
}

function isLegalMove(s, move) {
  const [r1, c1, r2, c2] = move;
  if (!inBounds(r1, c1) || !inBounds(r2, c2)) return false;
  if (s.grid[r1][c1] !== s.current_player) return false;
  if (s.grid[r2][c2] !== 0) return false;
  const d = chebyshev(r1, c1, r2, c2);
  return d === 1 || d === 2;
}

function getValidMoves(s) {
  const moves = [];
  for (let r1 = 0; r1 < BOARD_SIZE; r1 += 1) {
    for (let c1 = 0; c1 < BOARD_SIZE; c1 += 1) {
      if (s.grid[r1][c1] !== s.current_player) continue;
      for (let r2 = Math.max(0, r1 - 2); r2 <= Math.min(BOARD_SIZE - 1, r1 + 2); r2 += 1) {
        for (let c2 = Math.max(0, c1 - 2); c2 <= Math.min(BOARD_SIZE - 1, c1 + 2); c2 += 1) {
          if (isLegalMove(s, [r1, c1, r2, c2])) moves.push([r1, c1, r2, c2]);
        }
      }
    }
  }
  return moves;
}

function applyMove(s, move) {
  if (!move) {
    s.current_player *= -1;
    s.half_moves += 1;
    return;
  }
  const [r1, c1, r2, c2] = move;
  const d = chebyshev(r1, c1, r2, c2);
  s.grid[r2][c2] = s.current_player;
  if (d === 2) s.grid[r1][c1] = 0;
  for (let rr = Math.max(0, r2 - 1); rr <= Math.min(BOARD_SIZE - 1, r2 + 1); rr += 1) {
    for (let cc = Math.max(0, c2 - 1); cc <= Math.min(BOARD_SIZE - 1, c2 + 1); cc += 1) {
      if (s.grid[rr][cc] === -s.current_player) s.grid[rr][cc] = s.current_player;
    }
  }
  s.current_player *= -1;
  s.half_moves += 1;
}

function countPieces(s) {
  let p1 = 0;
  let p2 = 0;
  for (const row of s.grid) {
    for (const value of row) {
      if (value === HUMAN) p1 += 1;
      if (value === AI) p2 += 1;
    }
  }
  return { p1, p2 };
}

function render() {
  boardEl.innerHTML = "";
  for (let r = 0; r < BOARD_SIZE; r += 1) {
    for (let c = 0; c < BOARD_SIZE; c += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      if (selected && selected[0] === r && selected[1] === c) {
        cell.classList.add("selected");
      }
      cell.dataset.r = String(r);
      cell.dataset.c = String(c);
      const piece = state.grid[r][c];
      if (piece !== 0) {
        const dot = document.createElement("div");
        dot.className = `piece ${piece === HUMAN ? "p1" : "p2"}`;
        cell.appendChild(dot);
      }
      boardEl.appendChild(cell);
    }
  }
  const { p1, p2 } = countPieces(state);
  const turn = state.current_player === HUMAN ? "P1 (Human)" : "P2 (AI)";
  statusEl.textContent = thinking ? "Status: AI thinking..." : `Status: ${turn}`;
  scoreEl.textContent = `P1: ${p1} | P2: ${p2}`;
}

async function aiTurn() {
  if (thinking || state.current_player !== AI) return;
  thinking = true;
  render();
  try {
    const response = await fetch("/api/v1/gameplay/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board: state, mode: modeEl.value }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    const mv = payload.move;
    if (mv) applyMove(state, [mv.r1, mv.c1, mv.r2, mv.c2]);
    else applyMove(state, null);
  } catch (error) {
    statusEl.textContent = `Status: AI error (${String(error)})`;
  } finally {
    thinking = false;
    render();
  }
}

boardEl.addEventListener("click", async (event) => {
  const target = event.target.closest(".cell");
  if (!target || thinking || state.current_player !== HUMAN) return;
  const r = Number(target.dataset.r);
  const c = Number(target.dataset.c);
  if (!selected) {
    if (state.grid[r][c] === HUMAN) selected = [r, c];
    render();
    return;
  }
  if (state.grid[r][c] === HUMAN) {
    selected = [r, c];
    render();
    return;
  }
  const move = [selected[0], selected[1], r, c];
  if (isLegalMove(state, move)) {
    applyMove(state, move);
    selected = null;
    render();
    await aiTurn();
    return;
  }
  selected = null;
  render();
});

resetBtn.addEventListener("click", () => {
  state = makeInitialState();
  selected = null;
  thinking = false;
  render();
});

aiMoveBtn.addEventListener("click", async () => {
  await aiTurn();
});

render();
