from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame
import pygame.sndarray
import torch

if TYPE_CHECKING:
    from game.board import AtaxxBoard
    from model.system import AtaxxZero


Move = tuple[int, int, int, int]
Agent = str
PLAYER_1 = 1
PLAYER_2 = -1

CELL = 84
PAD = 24
SIDE = 290
BS = 7
BOARD_PX = BS * CELL
WIN_W = BOARD_PX + (2 * PAD) + SIDE
WIN_H = BOARD_PX + (2 * PAD)

AI_DELAY_MIN = 560
AI_DELAY_MAX = 1450
AI_DELAY_JITTER = 200
HIGHLIGHT_MS = 520
MOVE_PULSE_MS = 520
INFECT_PULSE_MS = 700
MOVE_PREVIEW_MS_HUMAN = 290
MOVE_PREVIEW_MS_AI = 470
INFECTION_STEP_MS = 100
SHAKE_MOVE_MS = 120
SHAKE_INFECT_MS = 180
SHAKE_MOVE_PX = 1.5
SHAKE_INFECT_PX = 2.6
FLASH_MOVE_MS = 90
FLASH_INFECT_MS = 170
PIECE_POP_MS = 200
PARTICLE_LIFE_MS = 340
PARTICLE_MAX_COUNT = 14
INTRO_STEP_MS = 520
END_FADE_MS = 650
END_COUNT_MS = 900
INTRO_STEPS = ("3", "2", "1", "FIGHT")

# UI theme: retro arcade + contemporary contrast
BG_TOP = (7, 10, 22)
BG_BOTTOM = (12, 20, 40)
BOARD_OUTER = (27, 34, 56)
TILE_A = (18, 24, 44)
TILE_B = (24, 31, 53)
GRID_LINE = (49, 61, 96)
PIECE_P1 = (255, 78, 110)
PIECE_P2 = (69, 180, 255)
PIECE_SHADOW = (5, 7, 12)
SELECTION = (61, 255, 158)
TARGET = (255, 215, 69)
RECENT = (255, 155, 68)
PREVIEW_MAIN = (84, 239, 255)
PREVIEW_GLOW = (153, 93, 255)
TEXT_MAIN = (236, 244, 255)
TEXT_DIM = (144, 164, 201)
PANEL_BG = (10, 14, 30)
PANEL_BORDER = (56, 74, 129)
PANEL_ACCENT = (115, 229, 255)
SCANLINE = (7, 12, 24)
FLASH_MOVE = (110, 245, 255)
FLASH_INFECT = (255, 150, 96)
TARGET_CLONE = (89, 255, 174)
TARGET_JUMP = (255, 196, 91)
HOVER_CELL = (156, 217, 255)
VIGNETTE = (4, 6, 16)

Particle = dict[str, float | tuple[int, int, int]]


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


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
        system = AtaxxZero.load_from_checkpoint(checkpoint_path, map_location=device)
    else:
        system = AtaxxZero()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid .pt checkpoint format")
            state_dict_obj = checkpoint.get("state_dict")
            if not isinstance(state_dict_obj, dict):
                raise ValueError("Checkpoint must contain key 'state_dict'")
            system.load_state_dict(state_dict_obj)
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

def _heuristic_move(board: AtaxxBoard, rng: np.random.Generator, level: str) -> Move | None:
    valid_moves = board.get_valid_moves()
    if len(valid_moves) == 0:
        return None

    def score_move(state: AtaxxBoard, move: Move) -> float:
        r1, c1, r2, c2 = move
        me = state.current_player
        before_me = int(np.sum(state.grid == me))
        before_opp = int(np.sum(state.grid == -me))
        scratch = state.copy()
        scratch.step(move)
        after_me = int(np.sum(scratch.grid == me))
        after_opp = int(np.sum(scratch.grid == -me))
        clone_bonus = 0.15 if max(abs(r1 - r2), abs(c1 - c2)) == 1 else 0.0
        center_bonus = 0.05 * (3 - abs(r2 - 3) + 3 - abs(c2 - 3))
        return float((after_me - before_me) + (before_opp - after_opp)) + clone_bonus + center_bonus

    if level == "easy":
        scores = np.asarray([score_move(board, move) for move in valid_moves], dtype=np.float32)
        scores = scores - float(np.min(scores)) + 0.2
        probs = scores / float(np.sum(scores))
        return valid_moves[int(rng.choice(len(valid_moves), p=probs))]

    scored_moves: list[tuple[Move, float]] = []
    for move in valid_moves:
        score = score_move(board, move)
        if level == "hard":
            scratch = board.copy()
            scratch.step(move)
            opp_moves = scratch.get_valid_moves()
            if len(opp_moves) > 0:
                opp_best = max(score_move(scratch, opp_move) for opp_move in opp_moves)
                score -= 0.65 * opp_best
        scored_moves.append((move, score))

    if level == "normal":
        # Normal: no es totalmente greedy. Muestreamos de forma suave por score
        # para evitar partidas clónicas y sesgo excesivo por desempates rígidos.
        scores = np.asarray([score for _, score in scored_moves], dtype=np.float32)
        temperature = 0.35
        logits = (scores - float(np.max(scores))) / temperature
        probs = np.exp(logits)
        probs = probs / float(np.sum(probs))
        pick_idx = int(rng.choice(len(scored_moves), p=probs))
        return scored_moves[pick_idx][0]

    best_score = max(score for _, score in scored_moves)
    best_moves = [move for move, score in scored_moves if score == best_score]
    return best_moves[int(rng.integers(0, len(best_moves)))]


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
    from engine.mcts import MCTS
    from game.actions import ACTION_SPACE

    if not board.has_valid_moves():
        return None, f"{agent} passed (no legal moves)"

    if agent == "random":
        moves = board.get_valid_moves()
        return moves[int(rng.integers(0, len(moves)))], "Random AI move played"
    if agent == "heuristic":
        return _heuristic_move(board, rng, heuristic_level), "Heuristic AI move played"
    if agent == "model":
        if not isinstance(mcts, MCTS):
            raise RuntimeError("Model agent selected but MCTS is not initialized.")
        probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
        action_idx = int(np.argmax(probs))
        return ACTION_SPACE.decode(action_idx), "Model AI move played"
    raise ValueError(f"Unsupported agent: {agent}")


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
    sample_rate = 44_100
    samples = int((duration_ms / 1000.0) * sample_rate)
    if samples <= 0:
        return None
    t = np.linspace(0, duration_ms / 1000.0, samples, endpoint=False)
    envelope = np.linspace(1.0, 0.1, samples)
    wave = np.sin(2 * np.pi * freq_hz * t) * envelope * volume
    stereo = np.stack([wave, wave], axis=1)
    audio_i16 = np.asarray(stereo * 32767.0, dtype=np.int16)
    return pygame.sndarray.make_sound(audio_i16)


def _build_sfx() -> dict[str, pygame.mixer.Sound | None]:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44_100, size=-16, channels=2, buffer=512)
    except pygame.error:
        return {"move": None, "infect": None}
    return {
        "move": _make_tone(300.0, 110, 0.08),
        "infect": _make_tone(160.0, 170, 0.10),
    }


def _play_sfx(sfx: dict[str, pygame.mixer.Sound | None], key: str) -> None:
    sound = sfx.get(key)
    if sound is not None:
        sound.play()


def _spawn_particles(
    rng: np.random.Generator,
    particles: list[Particle],
    cell: tuple[int, int],
    color: tuple[int, int, int],
    now_ms: int,
    intensity: int,
) -> None:
    rr, cc = cell
    cx = PAD + (cc * CELL) + (CELL // 2)
    cy = PAD + (rr * CELL) + (CELL // 2)
    count = int(min(PARTICLE_MAX_COUNT, max(5, intensity)))
    for _ in range(count):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(0.5, 2.3))
        particles.append(
            {
                "x": float(cx),
                "y": float(cy),
                "vx": float(np.cos(angle) * speed),
                "vy": float(np.sin(angle) * speed - 0.4),
                "start": float(now_ms),
                "end": float(now_ms + PARTICLE_LIFE_MS),
                "size": float(rng.uniform(1.8, 3.8)),
                "color": color,
            },
        )


def _wrap_text_line(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    if text == "":
        return [""]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            # Palabra muy larga: cortar por caracteres.
            chunk = ""
            for ch in word:
                test = chunk + ch
                if font.size(test)[0] <= max_width:
                    chunk = test
                else:
                    if chunk:
                        lines.append(chunk)
                    chunk = ch
            current = chunk
    if current:
        lines.append(current)
    return lines


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
    scene = pygame.Surface((WIN_W, WIN_H))

    for y in range(WIN_H):
        t = y / max(1, WIN_H - 1)
        color = (
            int(BG_TOP[0] * (1.0 - t) + BG_BOTTOM[0] * t),
            int(BG_TOP[1] * (1.0 - t) + BG_BOTTOM[1] * t),
            int(BG_TOP[2] * (1.0 - t) + BG_BOTTOM[2] * t),
        )
        pygame.draw.line(scene, color, (0, y), (WIN_W, y))

    for y in range(0, WIN_H, 4):
        pygame.draw.line(scene, SCANLINE, (0, y), (WIN_W, y), 1)

    brect = pygame.Rect(PAD, PAD, BOARD_PX, BOARD_PX)
    outer = brect.inflate(18, 18)
    pygame.draw.rect(scene, BOARD_OUTER, outer, border_radius=16)
    pygame.draw.rect(scene, PANEL_BORDER, outer, width=2, border_radius=16)
    board_fx = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    for i in range(3):
        rect = brect.inflate(26 + (i * 16), 26 + (i * 16))
        alpha = 34 - (i * 8)
        pygame.draw.rect(board_fx, (*PANEL_ACCENT, alpha), rect, width=2 + i, border_radius=20 + (i * 6))
    for i in range(4):
        rect = pygame.Rect(PAD + (i * 10), PAD + (i * 10), BOARD_PX - (i * 20), BOARD_PX - (i * 20))
        pygame.draw.rect(board_fx, (*VIGNETTE, 12 + (i * 8)), rect, width=9, border_radius=12)
    scene.blit(board_fx, (0, 0))

    for r in range(BS):
        for c in range(BS):
            tile = pygame.Rect(PAD + (c * CELL), PAD + (r * CELL), CELL, CELL)
            tile_color = TILE_A if (r + c) % 2 == 0 else TILE_B
            pygame.draw.rect(scene, tile_color, tile)
            pygame.draw.rect(scene, GRID_LINE, tile, width=1)

    for r in range(BS):
        for c in range(BS):
            cx = PAD + (c * CELL) + (CELL // 2)
            cy = PAD + (r * CELL) + (CELL // 2)
            cell_key = (r, c)
            if cell_key in infection_hidden and now_ms < infection_hidden[cell_key][0]:
                v = infection_hidden[cell_key][1]
            else:
                v = int(board.grid[r, c])
            scale = 1.0
            pop_span = piece_pop.get(cell_key)
            if pop_span is not None:
                pop_start, pop_end = pop_span
                if now_ms < pop_start:
                    scale = 0.55
                elif now_ms < pop_end:
                    progress = (now_ms - pop_start) / max(1, pop_end - pop_start)
                    ease = 1.0 - ((1.0 - progress) ** 2)
                    scale = 0.55 + (0.45 * ease)
            radius = int((CELL // 3) * scale)
            radius = max(8, radius)
            if v == PLAYER_1:
                pygame.draw.circle(scene, PIECE_SHADOW, (cx + 2, cy + 3), radius)
                pygame.draw.circle(scene, PIECE_P1, (cx, cy), radius)
                pygame.draw.circle(scene, (255, 170, 190), (cx - 8, cy - 8), 8)
            elif v == PLAYER_2:
                pygame.draw.circle(scene, PIECE_SHADOW, (cx + 2, cy + 3), radius)
                pygame.draw.circle(scene, PIECE_P2, (cx, cy), radius)
                pygame.draw.circle(scene, (175, 226, 255), (cx - 8, cy - 8), 8)

    if preview_move is not None and now_ms < preview_until:
        r1, c1, r2, c2 = preview_move
        sx = PAD + (c1 * CELL) + (CELL // 2)
        sy = PAD + (r1 * CELL) + (CELL // 2)
        tx = PAD + (c2 * CELL) + (CELL // 2)
        ty = PAD + (r2 * CELL) + (CELL // 2)
        pulse = 0.5 + (0.5 * np.sin(now_ms / 120))
        alpha_line = int(120 + (110 * pulse))
        glow_color = (*PREVIEW_GLOW, alpha_line)
        glow = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.line(glow, glow_color, (sx, sy), (tx, ty), 8)
        scene.blit(glow, (0, 0))
        pygame.draw.line(scene, PREVIEW_MAIN, (sx, sy), (tx, ty), 3)
        pygame.draw.circle(scene, TARGET, (tx, ty), 16, width=3)
        pygame.draw.circle(scene, PREVIEW_MAIN, (tx, ty), 6)
        if preview_started_at is not None and preview_until > preview_started_at:
            progress = (now_ms - preview_started_at) / (preview_until - preview_started_at)
            progress = float(max(0.0, min(1.0, progress)))
            eased = 1.0 - ((1.0 - progress) ** 2)
            hx = int(sx + ((tx - sx) * eased))
            hy = int(sy + ((ty - sy) * eased))
            pygame.draw.circle(scene, PREVIEW_MAIN, (hx, hy), 7)
            for t in (0.18, 0.36, 0.54):
                tail_t = max(0.0, eased - t)
                px = int(sx + ((tx - sx) * tail_t))
                py = int(sy + ((ty - sy) * tail_t))
                pygame.draw.circle(scene, PREVIEW_GLOW, (px, py), 3)

    if selected is not None:
        sr, sc = selected
        rect = pygame.Rect(PAD + (sc * CELL), PAD + (sr * CELL), CELL, CELL)
        pygame.draw.rect(scene, SELECTION, rect, width=4, border_radius=8)
    elif hover_cell is not None:
        hr, hc = hover_cell
        if int(board.grid[hr, hc]) == board.current_player:
            rect = pygame.Rect(PAD + (hc * CELL), PAD + (hr * CELL), CELL, CELL)
            pygame.draw.rect(scene, HOVER_CELL, rect, width=2, border_radius=8)

    targets_to_draw = legal_targets if selected is not None else hover_targets
    target_kinds = legal_target_kind if selected is not None else hover_target_kind
    hovered_target = hover_cell if selected is not None else None
    for tr, tc in targets_to_draw:
        cx = PAD + (tc * CELL) + (CELL // 2)
        cy = PAD + (tr * CELL) + (CELL // 2)
        kind = target_kinds.get((tr, tc), "jump")
        color = TARGET_CLONE if kind == "clone" else TARGET_JUMP
        radius = 11 if hovered_target == (tr, tc) else 9
        width = 3 if hovered_target == (tr, tc) else 2
        pygame.draw.circle(scene, color, (cx, cy), radius, width=width)
        pygame.draw.circle(scene, color, (cx, cy), 3)

    for rr, cc in recent:
        rect = pygame.Rect(PAD + (cc * CELL), PAD + (rr * CELL), CELL, CELL)
        pygame.draw.rect(scene, RECENT, rect, width=3, border_radius=8)

    if now_ms < move_until and len(move_cells) > 0:
        progress = 1.0 - ((move_until - now_ms) / MOVE_PULSE_MS)
        radius = int(10 + (progress * 22))
        for rr, cc in move_cells:
            cx = PAD + (cc * CELL) + (CELL // 2)
            cy = PAD + (rr * CELL) + (CELL // 2)
            pygame.draw.circle(scene, SELECTION, (cx, cy), radius, width=3)

    if now_ms < infect_until and len(infect_cells) > 0:
        progress = 1.0 - ((infect_until - now_ms) / INFECT_PULSE_MS)
        outer_radius = int(8 + (progress * 24))
        inner_radius = int(4 + (progress * 12))
        for rr, cc in infect_cells:
            cx = PAD + (cc * CELL) + (CELL // 2)
            cy = PAD + (rr * CELL) + (CELL // 2)
            pygame.draw.circle(scene, (255, 135, 84), (cx, cy), outer_radius, width=3)
            pygame.draw.circle(scene, (255, 228, 118), (cx, cy), inner_radius, width=2)

    panel_x = PAD + BOARD_PX + 20
    turn_text = (
        f"Turn: P1 (Red) [{p1_agent}]"
        if board.current_player == PLAYER_1
        else f"Turn: P2 (Blue) [{p2_agent}]"
    )
    mode_text = "Mode: spectate" if p1_agent != "human" and p2_agent != "human" else "Mode: play"
    if turn_agent != "human" and not board.is_game_over():
        dots = "." * ((now_ms // 260) % 4)
        think_text = f"{turn_agent} thinking{dots}"
    else:
        think_text = ""

    p1_count, p2_count = _counts(board)
    lines = [
        turn_text,
        mode_text,
        f"P1: {p1_count}  P2: {p2_count}",
        f"P1 agent: {p1_agent}",
        f"P2 agent: {p2_agent}",
        (f"Heuristic levels: P1={p1_level} | P2={p2_level}" if p1_level != "-" or p2_level != "-" else ""),
        status,
        think_text,
        "",
        "Controls:",
        "Click piece -> click target",
        "R: reset game",
        "Q: quit",
    ]

    panel_rect = pygame.Rect(panel_x - 12, PAD - 6, SIDE - 20, WIN_H - (2 * PAD) + 12)
    pygame.draw.rect(scene, PANEL_BG, panel_rect, border_radius=14)
    pygame.draw.rect(scene, PANEL_BORDER, panel_rect, width=2, border_radius=14)
    pygame.draw.line(
        scene,
        PANEL_ACCENT,
        (panel_rect.left + 12, panel_rect.top + 44),
        (panel_rect.right - 12, panel_rect.top + 44),
        2,
    )

    title = font.render("ATAXX ARENA", True, TEXT_MAIN)
    scene.blit(title, (panel_x, PAD))

    y = PAD + 54
    max_text_width = panel_rect.width - 24
    panel_bottom = panel_rect.bottom - 16
    for line in lines:
        wrapped = _wrap_text_line(small, line, max_text_width)
        if line == "":
            y += 10
            continue
        color = TEXT_MAIN if line == status or line.startswith("Turn:") else TEXT_DIM
        for wrapped_line in wrapped:
            if y > panel_bottom:
                break
            txt = small.render(wrapped_line, True, color)
            scene.blit(txt, (panel_x, y))
            y += 24
        if y > panel_bottom:
            break

    if board.is_game_over():
        overlay = font.render(_result_text(board), True, TEXT_MAIN)
        scene.blit(overlay, (panel_x, y + 8))

    if particles:
        pfx = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for p in particles:
            start_ms = float(p["start"])
            end_ms = float(p["end"])
            if now_ms < start_ms or now_ms >= end_ms:
                continue
            life = (now_ms - start_ms) / max(1.0, end_ms - start_ms)
            x = float(p["x"]) + (float(p["vx"]) * (now_ms - start_ms) * 0.08)
            y = float(p["y"]) + (float(p["vy"]) * (now_ms - start_ms) * 0.08) + (2.0 * life * life)
            alpha = int(220 * (1.0 - life))
            r, g, b = p["color"]
            size = max(1, int(float(p["size"]) * (1.0 - (0.4 * life))))
            pygame.draw.circle(pfx, (r, g, b, alpha), (int(x), int(y)), size)
        scene.blit(pfx, (0, 0))

    if now_ms < intro_until:
        intro_elapsed = now_ms - intro_start
        intro_idx = min(len(INTRO_STEPS) - 1, max(0, intro_elapsed // INTRO_STEP_MS))
        step_phase = (intro_elapsed % INTRO_STEP_MS) / INTRO_STEP_MS
        pulse = 0.25 + (0.75 * np.sin(np.pi * step_phase))
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((5, 8, 16, 170))
        scene.blit(overlay, (0, 0))
        text = INTRO_STEPS[int(intro_idx)]
        text_color = TARGET if text == "FIGHT" else PANEL_ACCENT
        title = big.render(text, True, text_color)
        scale = 0.88 + (0.18 * pulse)
        tw = int(title.get_width() * scale)
        th = int(title.get_height() * scale)
        title_scaled = pygame.transform.smoothscale(title, (max(1, tw), max(1, th)))
        tx = (WIN_W - title_scaled.get_width()) // 2
        ty = (WIN_H - title_scaled.get_height()) // 2
        scene.blit(title_scaled, (tx, ty))

    if game_over_started is not None and final_counts is not None:
        fade_progress = min(1.0, (now_ms - game_over_started) / END_FADE_MS)
        count_progress = min(1.0, (now_ms - game_over_started) / END_COUNT_MS)
        shown_p1 = int(final_counts[0] * count_progress)
        shown_p2 = int(final_counts[1] * count_progress)
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((4, 6, 12, int(170 * fade_progress)))
        scene.blit(overlay, (0, 0))
        card_w = 420
        card_h = 230
        card_x = (WIN_W - card_w) // 2
        card_y = (WIN_H - card_h) // 2
        card = pygame.Rect(card_x, card_y, card_w, card_h)
        pygame.draw.rect(scene, PANEL_BG, card, border_radius=16)
        pygame.draw.rect(scene, PANEL_BORDER, card, width=2, border_radius=16)
        head = font.render("MATCH RESULT", True, TEXT_MAIN)
        scene.blit(head, (card_x + 24, card_y + 18))
        result = font.render(_result_text(board), True, TARGET)
        scene.blit(result, (card_x + 24, card_y + 58))
        line = small.render(f"P1 (Red):  {shown_p1}", True, PIECE_P1)
        scene.blit(line, (card_x + 24, card_y + 106))
        line2 = small.render(f"P2 (Blue): {shown_p2}", True, PIECE_P2)
        scene.blit(line2, (card_x + 24, card_y + 136))
        hint = small.render("Press R to restart  |  Q to quit", True, TEXT_DIM)
        scene.blit(hint, (card_x + 24, card_y + 182))

    screen.fill((0, 0, 0))
    screen.blit(scene, shake_offset)

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
            if ai_ready_at[player] is None:
                ai_ready_at[player] = now_ms + _ai_delay_ms(board, turn_agent, args.mcts_sims, rng)
                status = f"{turn_agent} thinking..."
            elif now_ms >= int(ai_ready_at[player]):
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
