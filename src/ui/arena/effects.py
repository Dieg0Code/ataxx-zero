from __future__ import annotations

from typing import TypedDict

import numpy as np
import pygame
import pygame.sndarray


class Particle(TypedDict):
    x: float
    y: float
    vx: float
    vy: float
    start: float
    end: float
    size: float
    color: tuple[int, int, int]


def make_tone(freq_hz: float, duration_ms: int, volume: float = 0.12) -> pygame.mixer.Sound | None:
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


def build_sfx() -> dict[str, pygame.mixer.Sound | None]:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44_100, size=-16, channels=2, buffer=512)
    except pygame.error:
        return {"move": None, "infect": None}
    return {
        "move": make_tone(300.0, 110, 0.08),
        "infect": make_tone(160.0, 170, 0.10),
    }


def play_sfx(sfx: dict[str, pygame.mixer.Sound | None], key: str) -> None:
    sound = sfx.get(key)
    if sound is not None:
        sound.play()


def spawn_particles(
    rng: np.random.Generator,
    particles: list[Particle],
    cell: tuple[int, int],
    color: tuple[int, int, int],
    now_ms: int,
    intensity: int,
    *,
    pad: int,
    cell_px: int,
    particle_max_count: int,
    particle_life_ms: int,
) -> None:
    rr, cc = cell
    cx = pad + (cc * cell_px) + (cell_px // 2)
    cy = pad + (rr * cell_px) + (cell_px // 2)
    count = int(min(particle_max_count, max(5, intensity)))
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
                "end": float(now_ms + particle_life_ms),
                "size": float(rng.uniform(1.8, 3.8)),
                "color": color,
            },
        )


def wrap_text_line(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
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
            continue

        # Very long token: fallback to character slicing.
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
