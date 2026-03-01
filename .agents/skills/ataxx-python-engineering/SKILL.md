---
name: ataxx-python-engineering
description: End-to-end engineering workflow for the Ataxx Python monorepo (game engine, training, inference, UI, and API). Use when implementing or refactoring code in this repository and require strict quality gates with uv, Python 3.10+, TDD-first changes, and mandatory validation via ruff, pyrefly, and pytest before considering work done.
---

# Ataxx Python Engineering

Apply this workflow for all code changes in this repository.

## Workflow

1. Gather only required context.
   - If changing API/DB, read `references/api-structure.md` first.
   - If changing data models, ratings, replay, or training-data persistence, read `references/db-schema.md`.
2. Define the smallest behavior change and write/extend tests first.
3. Implement the change.
4. Run quality gates.
5. Report results, residual risks, and next step.

Keep edits incremental and reversible. Avoid large mixed refactors in one patch.

## Quality Gates (Mandatory)

Run these commands with `uv`:

```bash
uv run ruff check src tests scripts
uv run pyrefly check src tests
uv run pytest -q
```

If a change is local to one module, run module-scoped checks first, then run full checks before completion.

Do not finish with failing lint/type/tests.

## Project Rules

- Use Python 3.10+ compatible code.
- Use `uv` commands only (`uv run`, `uv sync`, `uv add`, `uv remove`).
- Prefer TDD:
  - add test exposing the behavior or regression;
  - implement fix;
  - validate all gates.
- Preserve existing architecture boundaries:
  - `src/game`: pure rules/state/actions
  - `src/model` and `src/engine`: training/search
  - `src/inference`: runtime move selection
  - `src/ui`: pygame client
  - `src/api`: backend service

## Educational Commenting Policy

This repository is educational, so comments must teach intent, not restate syntax.

- Add minimal, high-signal comments in non-trivial code paths.
- Prefer short comments that answer:
  - why this logic exists,
  - what invariant/assumption must hold,
  - what failure mode is being prevented.
- Do not add noise comments for obvious lines (e.g., "assign variable", "loop over list").
- Favor one good comment before a dense block over many inline comments.
- Keep comments resilient to refactors:
  - avoid mentioning transient details unless they are required constraints.
- When implementing algorithms (MCTS, training targets, replay shaping, serialization constraints),
  add at least one pedagogical comment describing the perspective/sign convention or invariant.
- For bug fixes, include a brief comment near the fix explaining the original failure mode.

## UI Contract (Web)

When changing `web/`, enforce a consistent design system and interaction model.

- Semantic action hierarchy:
  - `primary`: only for the main CTA in a section/screen.
  - `secondary`: default navigation and neutral actions.
  - `dangerSoft`: destructive entry action (e.g., "Eliminar" in lists).
  - `danger`: destructive confirmation action (e.g., modal confirm).
  - `ghost/tertiary`: low-emphasis utility actions.
- Never use `primary` styling for routine navigation actions like "Detalle".
- Prefer component variants over inline ad-hoc styling:
  - if a button/badge/card semantic exists, use it instead of custom one-off classes.
- Modal requirements:
  - themed overlay/background,
  - keyboard close via `Esc`,
  - click-away close,
  - body scroll lock while open,
  - clear destructive hierarchy (`Cancelar` vs confirm action).
- Status presentation:
  - do not encode state using color alone; include explicit text labels.
- Scrollbars in custom panels must use project-themed styles; avoid browser-default white scrollbars.
- Motion guidelines:
  - short transitions for hover/focus,
  - medium transitions for panel/modal enter-exit,
  - avoid noisy animation on non-critical UI.
- Design quality gate for UI PRs:
  - check button hierarchy on affected screens,
  - verify modal behavior and keyboard accessibility,
  - verify visual consistency with existing theme tokens in `web/src/shared/styles/`.

## Visual Language & Lore Guardrails

Use a consistent narrative and visual tone across the web UI.

- Core fantasy:
  - The board is the "hot zone" (highest visual intensity).
  - The opponent AI should feel dangerous and highly strategic.
  - Lore is suggested through atmosphere, not repeated literal text.
- Tone:
  - Minimal, tactical, high-signal UI.
  - Avoid excessive copy; prefer short labels and meaningful status.
  - Do not spam words like "alien", "malware", "IA", etc. in every section.
- Iconography-first communication:
  - Prefer icon + short label over long explanatory paragraphs.
  - Use icons to reinforce intent:
    - navigation/info (`Compass`, `Info`, `Clock`, `History`),
    - competition (`Trophy`, `Shield`, `Swords`),
    - danger/destructive (`AlertTriangle`, `Trash2`),
    - system/AI (`Cpu`, `Radar`, `Sparkles`).
  - Icons must not replace critical text entirely; keep compact accessible labels.
- Visual emphasis rules:
  - Board interactions and match-critical states can use richer motion/glow.
  - Surrounding pages (profile, ranking, landing) should remain calmer/subtler.
  - If something is not gameplay-critical, reduce visual intensity.
- Copy rules:
  - Spanish-first UX text.
  - Prefer concise, concrete microcopy.
  - Replace technical/internal wording with player-facing wording.
  - Keep status labels normalized and consistent across screens.

## Change Strategy

- Prefer small PR-like commits in spirit: one intent per change.
- Add regression tests for bug fixes, especially around:
  - move legality and pass logic,
  - action-space mapping,
  - checkpoint loading/inference modes,
  - script import path safety (`scripts/play_pygame.py`).
- Keep type hints strict enough to satisfy `pyrefly`.

## References

- Quality gates and per-domain command matrix: `references/quality-gates.md`
- Repository structure and fragile areas: `references/project-map.md`
- API target structure and module boundaries: `references/api-structure.md`
- Current and planned database schema: `references/db-schema.md`
- Migration discipline (Alembic workflow): `references/migration-discipline.md`
- Canonical uv command snippets: `references/commands.md`
