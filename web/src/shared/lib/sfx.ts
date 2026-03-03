type SfxPool = {
  elements: HTMLAudioElement[];
  nextIndex: number;
};

const POOLS = new Map<string, SfxPool>();
const BUFFER_CACHE = new Map<string, AudioBuffer>();
const BUFFER_LOADS = new Map<string, Promise<void>>();
let primedOnInteraction = false;
let audioContext: AudioContext | null = null;

function canUseAudio(): boolean {
  return typeof window !== "undefined" && typeof Audio !== "undefined";
}

function buildAudioElement(path: string): HTMLAudioElement {
  const audio = new Audio(path);
  audio.preload = "auto";
  return audio;
}

function getAudioContext(): AudioContext | null {
  if (typeof window === "undefined") {
    return null;
  }
  const ContextCtor = window.AudioContext ?? (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (ContextCtor === undefined) {
    return null;
  }
  if (audioContext === null) {
    audioContext = new ContextCtor({ latencyHint: "interactive" });
  }
  return audioContext;
}

async function primeBuffer(path: string): Promise<void> {
  if (BUFFER_CACHE.has(path)) {
    return;
  }
  const inFlight = BUFFER_LOADS.get(path);
  if (inFlight !== undefined) {
    await inFlight;
    return;
  }
  const context = getAudioContext();
  if (context === null) {
    return;
  }
  const task = (async () => {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`SFX fetch failed (${response.status}) for ${path}`);
    }
    const payload = await response.arrayBuffer();
    const decoded = await context.decodeAudioData(payload.slice(0));
    BUFFER_CACHE.set(path, decoded);
  })()
    .catch(() => {
      // Keep fallback path via HTMLAudio channels.
    })
    .finally(() => {
      BUFFER_LOADS.delete(path);
    });
  BUFFER_LOADS.set(path, task);
  await task;
}

function playBuffered(path: string, volume: number): boolean {
  const context = getAudioContext();
  if (context === null) {
    return false;
  }
  const buffer = BUFFER_CACHE.get(path);
  if (buffer === undefined) {
    return false;
  }
  if (context.state === "suspended") {
    void context.resume();
    return false;
  }
  const gain = context.createGain();
  gain.gain.value = Math.max(0, Math.min(1, volume));
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(gain);
  gain.connect(context.destination);
  source.start(0);
  return true;
}

function ensurePool(path: string, poolSize: number): SfxPool | null {
  if (!canUseAudio()) {
    return null;
  }
  const existing = POOLS.get(path);
  if (existing !== undefined) {
    return existing;
  }
  const safePoolSize = Math.max(1, poolSize);
  const elements = Array.from({ length: safePoolSize }, () => buildAudioElement(path));
  const pool: SfxPool = {
    elements,
    nextIndex: 0,
  };
  POOLS.set(path, pool);
  return pool;
}

function pickAudioChannel(pool: SfxPool): HTMLAudioElement {
  const idle = pool.elements.find((item) => item.paused || item.ended);
  if (idle !== undefined) {
    return idle;
  }
  const element = pool.elements[pool.nextIndex];
  pool.nextIndex = (pool.nextIndex + 1) % pool.elements.length;
  return element;
}

export function primeSfx(paths: readonly string[], poolSize = 4): void {
  if (!canUseAudio()) {
    return;
  }
  for (const path of paths) {
    ensurePool(path, poolSize);
    void primeBuffer(path);
  }
}

export function primeSfxOnFirstInteraction(paths: readonly string[], poolSize = 4): void {
  if (!canUseAudio() || primedOnInteraction) {
    return;
  }
  const primeOnce = (): void => {
    const context = getAudioContext();
    if (context !== null && context.state === "suspended") {
      void context.resume();
    }
    primeSfx(paths, poolSize);
    window.removeEventListener("pointerdown", primeOnce);
    window.removeEventListener("pointermove", primeOnce);
    window.removeEventListener("touchstart", primeOnce);
    window.removeEventListener("keydown", primeOnce);
  };
  primedOnInteraction = true;
  window.addEventListener("pointerdown", primeOnce, { once: true });
  window.addEventListener("pointermove", primeOnce, { once: true, passive: true });
  window.addEventListener("touchstart", primeOnce, { once: true });
  window.addEventListener("keydown", primeOnce, { once: true });
}

export function playSfx(path: string, volume = 0.3): void {
  if (!canUseAudio()) {
    return;
  }
  try {
    if (playBuffered(path, volume)) {
      return;
    }
    // Reusing decoded channels removes first-play latency and hover lag.
    const pool = ensurePool(path, 4);
    if (pool === null) {
      return;
    }
    const audio = pickAudioChannel(pool);
    audio.volume = Math.max(0, Math.min(1, volume));
    audio.currentTime = 0;
    const playResult = audio.play();
    if (playResult && typeof playResult.catch === "function") {
      void playResult.catch(() => {});
    }
  } catch {
    // Ignore browser/runtime audio errors in unsupported environments.
  }
}
