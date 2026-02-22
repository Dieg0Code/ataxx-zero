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
