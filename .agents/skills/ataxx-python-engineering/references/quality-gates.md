# Quality Gates

Use these checks before completing work.

## Global

```bash
uv run ruff check src tests scripts
uv run pyrefly check src tests
uv run pytest -q
```

## Domain-focused quick checks

### API and DB

```bash
uv run ruff check src/api tests
uv run pyrefly check src/api tests
uv run pytest -q
```

### Game / Engine / Inference

```bash
uv run ruff check src/game src/engine src/inference tests
uv run pyrefly check src/game src/engine src/inference tests
uv run pytest -q
```

### UI / Scripts

```bash
uv run ruff check src/ui scripts tests
uv run pyrefly check src/ui scripts
uv run pytest -q
```

## Definition of Done

- Lint clean.
- Type check clean.
- Tests green.
- New behavior covered by tests.
