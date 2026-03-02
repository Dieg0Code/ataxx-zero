from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame
import pygame.sndarray
import torch

# Ensure `src/` is importable when running as:
# `uv run python scripts/play_pygame.py ...`
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ui.arena.layout import BOARD_PX, CELL, PAD, WIN_H, WIN_W  # noqa: E402
from ui.arena.theme import (  # noqa: E402
    AI_DELAY_JITTER,
    AI_DELAY_MAX,
    AI_DELAY_MIN,
    FLASH_INFECT,
    FLASH_INFECT_MS,
    FLASH_MOVE,
    FLASH_MOVE_MS,
    HIGHLIGHT_MS,
    INFECT_PULSE_MS,
    INFECTION_STEP_MS,
    INTRO_STEP_MS,
    INTRO_STEPS,
    MOVE_PREVIEW_MS_AI,
    MOVE_PREVIEW_MS_HUMAN,
    MOVE_PULSE_MS,
    PARTICLE_LIFE_MS,
    PARTICLE_MAX_COUNT,
    PIECE_POP_MS,
    SHAKE_INFECT_MS,
    SHAKE_INFECT_PX,
    SHAKE_MOVE_MS,
    SHAKE_MOVE_PX,
)

if TYPE_CHECKING:
    from agents.types import Agent
    from game.board import AtaxxBoard
    from game.types import Move
    from model.system import AtaxxZero
    from ui.arena.effects import Particle

PLAYER_1 = 1
PLAYER_2 = -1

def _ensure_src_on_path() -> None:
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ataxx arena (play/spectate).")
    parser.add_argument("--mode", default="play", choices=["play", "spectate"])
    parser.add_argument("--checkpoint", "--ckpt", default="")
    parser.add_argument("--opponent", "--opp", default="heuristic", choices=["random", "heuristic", "model"])
    parser.add_argument("--human-player", "--human-side", default="p1", choices=["p1", "p2"])
    parser.add_argument(
        "--p1-agent",
        "--agent1",
        default="",
        choices=["", "human", "random", "heuristic", "model"],
    )
    parser.add_argument(
        "--p2-agent",
        "--agent2",
        default="",
        choices=["", "human", "random", "heuristic", "model"],
    )
    parser.add_argument("--heuristic-level", "--level", default="normal", choices=["easy", "normal", "hard"])
    parser.add_argument("--p1-level", "--level1", default="", choices=["", "easy", "normal", "hard"])
    parser.add_argument("--p2-level", "--level2", default="", choices=["", "easy", "normal", "hard"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mcts-sims", "--sims", type=int, default=160)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="RNG seed. Use -1 for non-deterministic runs.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _resolve_agents(args: argparse.Namespace) -> tuple[Agent, Agent]:
    if args.p1_agent or args.p2_agent:
        p1 = args.p1_agent or "heuristic"
        p2 = args.p2_agent or "heuristic"
    elif args.mode == "spectate":
        p1, p2 = "heuristic", "heuristic"
    elif args.human_player == "p1":
        p1, p2 = "human", args.opponent
    else:
        p1, p2 = args.opponent, "human"

    if args.mode == "spectate" and (p1 == "human" or p2 == "human"):
        raise ValueError("spectate mode requires AI vs AI")
    if args.mode == "play" and p1 != "human" and p2 != "human":
        raise ValueError("play mode requires at least one human")
    return p1, p2


def _resolve_heuristic_levels(
    args: argparse.Namespace,
    p1_agent: Agent,
    p2_agent: Agent,
) -> tuple[str, str]:
    default_level = args.heuristic_level
    p1_level = args.p1_level or default_level
    p2_level = args.p2_level or default_level
    if p1_agent != "heuristic":
        p1_level = "-"
    if p2_agent != "heuristic":
        p2_level = "-"
    return p1_level, p2_level


def _load_system(checkpoint_path: str, device: str) -> AtaxxZero:
    from model.system import AtaxxZero

    if checkpoint_path.endswith(".ckpt"):
        try:
            system = AtaxxZero.load_from_checkpoint(checkpoint_path, map_location=device)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint incompatible con architecture policy_head espacial; "
                "reentrena o usa carga parcial manual (strict=False)."
            ) from exc
    else:
        system = AtaxxZero()
        if checkpoint_path:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid .pt checkpoint format")
            state_dict_obj = checkpoint.get("state_dict")
            if not isinstance(state_dict_obj, dict):
                raise ValueError("Checkpoint must contain key 'state_dict'")
            try:
                system.load_state_dict(state_dict_obj)
            except RuntimeError as exc:
                raise ValueError(
                    "Checkpoint incompatible con architecture policy_head espacial; "
                    "reentrena o usa carga parcial manual (strict=False)."
                ) from exc
    system.eval()
    system.to(device)
    return system


def _cell_from_pos(mx: int, my: int) -> tuple[int, int] | None:
    bx = mx - PAD
    by = my - PAD
    if bx < 0 or by < 0 or bx >= BOARD_PX or by >= BOARD_PX:
        return None
    return int(by // CELL), int(bx // CELL)


def _moves_from_origin(board: AtaxxBoard, origin: tuple[int, int]) -> list[Move]:
    r0, c0 = origin
    return [m for m in board.get_valid_moves() if m[0] == r0 and m[1] == c0]


def _targets_and_kinds(moves: list[Move]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]]:
    targets: list[tuple[int, int]] = []
    kinds: dict[tuple[int, int], str] = {}
    for r1, c1, r2, c2 in moves:
        target = (r2, c2)
        targets.append(target)
        dist = max(abs(r1 - r2), abs(c1 - c2))
        kinds[target] = "clone" if dist == 1 else "jump"
    return targets, kinds


def _counts(board: AtaxxBoard) -> tuple[int, int]:
    return int(np.sum(board.grid == PLAYER_1)), int(np.sum(board.grid == PLAYER_2))


def _result_text(board: AtaxxBoard) -> str:
    result = board.get_result()
    if result == 1:
        return "Winner: P1 (Red)"
    if result == -1:
        return "Winner: P2 (Blue)"
    return "Result: Draw"

def _ai_delay_ms(board: AtaxxBoard, agent: Agent, sims: int, rng: np.random.Generator) -> int:
    valid_count = len(board.get_valid_moves())
    complexity = min(260, valid_count * 9)
    model_bonus = 120 if agent == "model" else 0
    sims_bonus = min(180, sims // 2) if agent == "model" else 0
    jitter = int(rng.integers(0, AI_DELAY_JITTER + 1))
    delay = AI_DELAY_MIN + complexity + model_bonus + sims_bonus + jitter
    return int(max(AI_DELAY_MIN, min(AI_DELAY_MAX, delay)))


def _changed_cells(before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(before != after)
    return [(int(r), int(c)) for r, c in coords]


def _apply_move_with_feedback(
    board: AtaxxBoard,
    move: Move | None,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    tuple[int, int] | None,
]:
    before = board.grid.copy()
    player = board.current_player
    board.step(move)
    changed = _changed_cells(before, board.grid)

    move_cells: list[tuple[int, int]] = []
    infect_cells: list[tuple[int, int]] = []
    old_infection_values: dict[tuple[int, int], int] = {}
    destination: tuple[int, int] | None = None
    if move is not None:
        r1, c1, r2, c2 = move
        destination = (r2, c2)
        move_cells.append((r2, c2))
        if max(abs(r1 - r2), abs(c1 - c2)) == 2:
            move_cells.append((r1, c1))
        move_set = set(move_cells)
        for rr, cc in changed:
            if (rr, cc) in move_set:
                continue
            if int(before[rr, cc]) == -player and int(board.grid[rr, cc]) == player:
                infect_cells.append((rr, cc))
                old_infection_values[(rr, cc)] = int(before[rr, cc])
    return changed, move_cells, infect_cells, old_infection_values, destination


def _pick_ai_move(
    board: AtaxxBoard,
    agent: Agent,
    rng: np.random.Generator,
    heuristic_level: str,
    mcts: object | None,
) -> tuple[Move | None, str]:
    from agents.selector import pick_ai_move

    return pick_ai_move(
        board=board,
        agent=agent,
        rng=rng,
        heuristic_level=heuristic_level,
        mcts=mcts,
    )


def _infection_wave_schedule(
    destination: tuple[int, int] | None,
    infection_cells: list[tuple[int, int]],
    start_ms: int,
) -> dict[tuple[int, int], int]:
    if destination is None or len(infection_cells) == 0:
        return {}
    dr, dc = destination

    def sort_key(cell: tuple[int, int]) -> tuple[float, int]:
        rr, cc = cell
        angle = np.arctan2(rr - dr, cc - dc)
        radius = abs(rr - dr) + abs(cc - dc)
        return (float(angle), radius)

    ordered = sorted(infection_cells, key=sort_key)
    return {
        cell: start_ms + (idx * INFECTION_STEP_MS)
        for idx, cell in enumerate(ordered)
    }


def _make_tone(freq_hz: float, duration_ms: int, volume: float = 0.12) -> pygame.mixer.Sound | None:
    from ui.arena.effects import make_tone

    return make_tone(freq_hz=freq_hz, duration_ms=duration_ms, volume=volume)


def _build_sfx() -> dict[str, pygame.mixer.Sound | None]:
    from ui.arena.effects import build_sfx

    return build_sfx()


def _play_sfx(sfx: dict[str, pygame.mixer.Sound | None], key: str) -> None:
    from ui.arena.effects import play_sfx

    play_sfx(sfx=sfx, key=key)


def _spawn_particles(
    rng: np.random.Generator,
    particles: list[Particle],
    cell: tuple[int, int],
    color: tuple[int, int, int],
    now_ms: int,
    intensity: int,
) -> None:
    from ui.arena.effects import spawn_particles

    spawn_particles(
        rng=rng,
        particles=particles,
        cell=cell,
        color=color,
        now_ms=now_ms,
        intensity=intensity,
        pad=PAD,
        cell_px=CELL,
        particle_max_count=PARTICLE_MAX_COUNT,
        particle_life_ms=PARTICLE_LIFE_MS,
    )


def _wrap_text_line(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    from ui.arena.effects import wrap_text_line

    return wrap_text_line(font=font, text=text, max_width=max_width)


def _draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    big: pygame.font.Font,
    board: AtaxxBoard,
    selected: tuple[int, int] | None,
    legal_targets: list[tuple[int, int]],
    legal_target_kind: dict[tuple[int, int], str],
    hover_cell: tuple[int, int] | None,
    hover_targets: list[tuple[int, int]],
    hover_target_kind: dict[tuple[int, int], str],
    p1_agent: Agent,
    p2_agent: Agent,
    turn_agent: Agent,
    status: str,
    p1_level: str,
    p2_level: str,
    recent: list[tuple[int, int]],
    move_cells: list[tuple[int, int]],
    infect_cells: list[tuple[int, int]],
    infection_hidden: dict[tuple[int, int], tuple[int, int]],
    preview_move: Move | None,
    preview_started_at: int | None,
    preview_until: int,
    now_ms: int,
    move_until: int,
    infect_until: int,
    shake_offset: tuple[int, int],
    flash_start: int,
    flash_until: int,
    flash_color: tuple[int, int, int],
    piece_pop: dict[tuple[int, int], tuple[int, int]],
    particles: list[Particle],
    intro_start: int,
    intro_until: int,
    game_over_started: int | None,
    final_counts: tuple[int, int] | None,
) -> None:
    from ui.arena.render import draw_arena

    draw_arena(
        screen=screen,
        font=font,
        small=small,
        big=big,
        board=board,
        selected=selected,
        legal_targets=legal_targets,
        legal_target_kind=legal_target_kind,
        hover_cell=hover_cell,
        hover_targets=hover_targets,
        hover_target_kind=hover_target_kind,
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        turn_agent=turn_agent,
        status=status,
        p1_level=p1_level,
        p2_level=p2_level,
        recent=recent,
        move_cells=move_cells,
        infect_cells=infect_cells,
        infection_hidden=infection_hidden,
        preview_move=preview_move,
        preview_started_at=preview_started_at,
        preview_until=preview_until,
        now_ms=now_ms,
        move_until=move_until,
        infect_until=infect_until,
        shake_offset=shake_offset,
        flash_start=flash_start,
        flash_until=flash_until,
        flash_color=flash_color,
        piece_pop=piece_pop,
        particles=particles,
        intro_start=intro_start,
        intro_until=intro_until,
        game_over_started=game_over_started,
        final_counts=final_counts,
    )
def main() -> None:
    _ensure_src_on_path()
    from engine.mcts import MCTS
    from game.board import AtaxxBoard

    args = _parse_args()
    device = _resolve_device(args.device)
    p1_agent, p2_agent = _resolve_agents(args)
    p1_level, p2_level = _resolve_heuristic_levels(args, p1_agent, p2_agent)
    rng = np.random.default_rng(seed=None if args.seed < 0 else args.seed)

    mcts: MCTS | None = None
    if p1_agent == "model" or p2_agent == "model":
        system = _load_system(args.checkpoint, device=device)
        mcts = MCTS(
            model=system.model,
            c_puct=args.c_puct,
            n_simulations=args.mcts_sims,
            device=device,
        )

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Ataxx Arena")
    font = pygame.font.SysFont("consolas", 28)
    small = pygame.font.SysFont("consolas", 20)
    big = pygame.font.SysFont("consolas", 104, bold=True)
    clock = pygame.time.Clock()

    board = AtaxxBoard()
    selected: tuple[int, int] | None = None
    legal_targets: list[tuple[int, int]] = []
    legal_target_kind: dict[tuple[int, int], str] = {}
    hover_targets: list[tuple[int, int]] = []
    hover_target_kind: dict[tuple[int, int], str] = {}
    hover_cell: tuple[int, int] | None = None
    status = "Ready"
    recent: list[tuple[int, int]] = []
    recent_until = 0
    move_cells: list[tuple[int, int]] = []
    move_until = 0
    infect_cells: list[tuple[int, int]] = []
    infect_until = 0
    infection_hidden: dict[tuple[int, int], tuple[int, int]] = {}
    preview_move: Move | None = None
    preview_started_at: int | None = None
    preview_until = 0
    pending_move: Move | None = None
    pending_apply_at: int | None = None
    ai_ready_at: dict[int, int | None] = {PLAYER_1: None, PLAYER_2: None}
    shake_start = 0
    shake_until = 0
    shake_max_px = 0.0
    flash_start = 0
    flash_until = 0
    flash_color = FLASH_MOVE
    piece_pop: dict[tuple[int, int], tuple[int, int]] = {}
    particles: list[Particle] = []
    sfx = _build_sfx()
    intro_start = pygame.time.get_ticks()
    intro_until = intro_start + (INTRO_STEP_MS * len(INTRO_STEPS))
    game_over_started: int | None = None
    final_counts: tuple[int, int] | None = None

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        if recent and now_ms >= recent_until:
            recent = []
        if particles:
            particles = [p for p in particles if now_ms < float(p["end"])]
        if piece_pop:
            piece_pop = {cell: span for cell, span in piece_pop.items() if now_ms < span[1]}
        if infection_hidden:
            revealed_cells: list[tuple[int, int]] = [
                cell for cell, data in infection_hidden.items() if now_ms >= data[0]
            ]
            infection_hidden = {
                cell: data
                for cell, data in infection_hidden.items()
                if now_ms < data[0]
            }
            for cell in revealed_cells:
                piece_pop[cell] = (now_ms, now_ms + PIECE_POP_MS)
                _spawn_particles(rng, particles, cell, FLASH_INFECT, now_ms, intensity=8)

        turn_agent = p1_agent if board.current_player == PLAYER_1 else p2_agent
        intro_active = now_ms < intro_until
        if board.is_game_over() and game_over_started is None:
            game_over_started = now_ms
            final_counts = _counts(board)
        human_turn = turn_agent == "human" and not board.is_game_over() and not intro_active
        ai_turn = turn_agent != "human" and not board.is_game_over() and not intro_active
        hover_cell = None
        hover_targets = []
        hover_target_kind = {}
        if human_turn and pending_apply_at is None:
            hover_cell = _cell_from_pos(*pygame.mouse.get_pos())
            if selected is None and hover_cell is not None:
                hr, hc = hover_cell
                if int(board.grid[hr, hc]) == board.current_player:
                    hmoves = _moves_from_origin(board, hover_cell)
                    hover_targets, hover_target_kind = _targets_and_kinds(hmoves)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    board = AtaxxBoard()
                    selected = None
                    legal_targets = []
                    legal_target_kind = {}
                    hover_targets = []
                    hover_target_kind = {}
                    hover_cell = None
                    status = "Game reset"
                    ai_ready_at = {PLAYER_1: None, PLAYER_2: None}
                    recent = []
                    move_cells = []
                    infect_cells = []
                    infection_hidden = {}
                    preview_move = None
                    preview_started_at = None
                    preview_until = 0
                    pending_move = None
                    pending_apply_at = None
                    shake_start = 0
                    shake_until = 0
                    shake_max_px = 0.0
                    flash_start = 0
                    flash_until = 0
                    flash_color = FLASH_MOVE
                    piece_pop = {}
                    particles = []
                    intro_start = now_ms
                    intro_until = intro_start + (INTRO_STEP_MS * len(INTRO_STEPS))
                    game_over_started = None
                    final_counts = None
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and human_turn
                and pending_apply_at is None
            ):
                cell = _cell_from_pos(*pygame.mouse.get_pos())
                if cell is None:
                    continue
                row, col = cell
                if selected is None:
                    if int(board.grid[row, col]) == board.current_player:
                        selected = (row, col)
                        moves = _moves_from_origin(board, selected)
                        legal_targets, legal_target_kind = _targets_and_kinds(moves)
                else:
                    if (row, col) == selected:
                        selected = None
                        legal_targets = []
                        legal_target_kind = {}
                        continue
                    if int(board.grid[row, col]) == board.current_player:
                        selected = (row, col)
                        moves = _moves_from_origin(board, selected)
                        legal_targets, legal_target_kind = _targets_and_kinds(moves)
                        continue
                    candidate: Move | None = None
                    for move in _moves_from_origin(board, selected):
                        if move[2] == row and move[3] == col:
                            candidate = move
                            break
                    if candidate is not None:
                        selected = None
                        legal_targets = []
                        legal_target_kind = {}
                        pending_move = candidate
                        preview_started_at = now_ms
                        pending_apply_at = now_ms + MOVE_PREVIEW_MS_HUMAN
                        preview_move = candidate
                        preview_until = int(pending_apply_at)
                        status = "Preparing human move..."
                        ai_ready_at = {PLAYER_1: None, PLAYER_2: None}

        if human_turn and not board.has_valid_moves() and pending_apply_at is None:
            pending_move = None
            preview_started_at = now_ms
            pending_apply_at = now_ms + MOVE_PREVIEW_MS_HUMAN
            preview_move = None
            preview_until = int(pending_apply_at)
            status = "Human passing..."
            ai_ready_at = {PLAYER_1: None, PLAYER_2: None}

        if ai_turn and pending_apply_at is None:
            player = board.current_player
            ready_at = ai_ready_at[player]
            if ready_at is None:
                ai_ready_at[player] = now_ms + _ai_delay_ms(board, turn_agent, args.mcts_sims, rng)
                status = f"{turn_agent} thinking..."
            elif now_ms >= ready_at:
                move, move_text = _pick_ai_move(
                    board=board,
                    agent=turn_agent,
                    rng=rng,
                    heuristic_level=p1_level if player == PLAYER_1 else p2_level,
                    mcts=mcts,
                )
                pending_move = move
                preview_started_at = now_ms
                pending_apply_at = now_ms + MOVE_PREVIEW_MS_AI
                preview_move = move
                preview_until = int(pending_apply_at)
                status = move_text.replace("played", "queued")
                ai_ready_at = {PLAYER_1: None, PLAYER_2: None}
            else:
                status = f"{turn_agent} thinking..."

        if pending_apply_at is not None and now_ms >= pending_apply_at:
            changed, move_cells, infect_cells, old_vals, destination = _apply_move_with_feedback(
                board,
                pending_move,
            )
            recent = changed
            recent_until = now_ms + HIGHLIGHT_MS
            move_until = now_ms + MOVE_PULSE_MS
            infect_until = now_ms + INFECT_PULSE_MS

            reveal_schedule = _infection_wave_schedule(
                destination=destination,
                infection_cells=infect_cells,
                start_ms=now_ms + 40,
            )
            infection_hidden = {
                cell: (reveal_schedule.get(cell, now_ms), old_val)
                for cell, old_val in old_vals.items()
            }

            pending_move = None
            pending_apply_at = None
            preview_move = None
            preview_started_at = None
            preview_until = 0
            status = "Move resolved"

            has_infection = len(infect_cells) > 0
            shake_start = now_ms
            shake_until = now_ms + (SHAKE_INFECT_MS if has_infection else SHAKE_MOVE_MS)
            shake_max_px = SHAKE_INFECT_PX if has_infection else SHAKE_MOVE_PX
            flash_start = now_ms
            flash_until = now_ms + (FLASH_INFECT_MS if has_infection else FLASH_MOVE_MS)
            flash_color = FLASH_INFECT if has_infection else FLASH_MOVE
            _play_sfx(sfx, "infect" if has_infection else "move")

            for rr, cc in move_cells:
                if int(board.grid[rr, cc]) != 0:
                    piece_pop[(rr, cc)] = (now_ms, now_ms + PIECE_POP_MS)
                    _spawn_particles(rng, particles, (rr, cc), FLASH_MOVE, now_ms, intensity=10)

        shake_offset = (0, 0)
        if shake_until > now_ms:
            shake_progress = (now_ms - shake_start) / max(1, shake_until - shake_start)
            amp = shake_max_px * (1.0 - shake_progress)
            sx = int(np.sin(now_ms * 0.040) * amp)
            sy = int(np.cos(now_ms * 0.033) * (amp * 0.7))
            shake_offset = (sx, sy)

        _draw(
            screen=screen,
            font=font,
            small=small,
            big=big,
            board=board,
            selected=selected,
            legal_targets=legal_targets,
            legal_target_kind=legal_target_kind,
            hover_cell=hover_cell,
            hover_targets=hover_targets,
            hover_target_kind=hover_target_kind,
            p1_agent=p1_agent,
            p2_agent=p2_agent,
            turn_agent=turn_agent,
            status=status,
            p1_level=p1_level,
            p2_level=p2_level,
            recent=recent,
            move_cells=move_cells,
            infect_cells=infect_cells,
            infection_hidden=infection_hidden,
            preview_move=preview_move,
            preview_started_at=preview_started_at,
            preview_until=preview_until,
            now_ms=now_ms,
            move_until=move_until,
            infect_until=infect_until,
            shake_offset=shake_offset,
            flash_start=flash_start,
            flash_until=flash_until,
            flash_color=flash_color,
            piece_pop=piece_pop,
            particles=particles,
            intro_start=intro_start,
            intro_until=intro_until,
            game_over_started=game_over_started,
            final_counts=final_counts,
        )
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()


