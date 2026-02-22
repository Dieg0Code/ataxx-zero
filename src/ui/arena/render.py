from __future__ import annotations

import numpy as np
import pygame

from agents.types import Agent
from game.board import AtaxxBoard
from game.types import Move
from ui.arena.effects import Particle, wrap_text_line
from ui.arena.layout import BOARD_PX, BS, CELL, PAD, SIDE, WIN_H, WIN_W
from ui.arena.theme import (
    BG_BOTTOM,
    BG_TOP,
    BOARD_OUTER,
    END_COUNT_MS,
    END_FADE_MS,
    GRID_LINE,
    HOVER_CELL,
    INFECT_PULSE_MS,
    INTRO_STEP_MS,
    INTRO_STEPS,
    MOVE_PULSE_MS,
    PANEL_ACCENT,
    PANEL_BG,
    PANEL_BORDER,
    PIECE_P1,
    PIECE_P2,
    PIECE_SHADOW,
    PREVIEW_GLOW,
    PREVIEW_MAIN,
    RECENT,
    SCANLINE,
    SELECTION,
    TARGET,
    TARGET_CLONE,
    TARGET_JUMP,
    TEXT_DIM,
    TEXT_MAIN,
    TILE_A,
    TILE_B,
    VIGNETTE,
)

PLAYER_1 = 1
PLAYER_2 = -1
EMPTY = 0


def _counts(board: AtaxxBoard) -> tuple[int, int]:
    return int(np.sum(board.grid == PLAYER_1)), int(np.sum(board.grid == PLAYER_2))


def _result_text(board: AtaxxBoard) -> str:
    result = board.get_result()
    if result == 1:
        return "Winner: P1 (Red)"
    if result == -1:
        return "Winner: P2 (Blue)"
    return "Result: Draw"


def draw_arena(
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
    del flash_start, flash_until, flash_color

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
        wrapped = wrap_text_line(small, line, max_text_width)
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
