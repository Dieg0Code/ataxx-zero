# Commands

## Environment

```bash
uv sync --all-groups
```

If using partial groups, remember it can uninstall packages from omitted groups:

```bash
uv sync --group api --group dev
uv sync --group ui --group api --group dev
```

## Add/remove dependencies

```bash
uv add <package>
uv add --group api <package>
uv remove <package>
```

## Quality

```bash
uv run ruff check src tests scripts
uv run pyrefly check src tests
uv run pytest -q
```

## Targeted checks

```bash
uv run ruff check src/api
uv run pyrefly check src/api
uv run pytest -q tests/test_inference_service.py
```

## Training / UI examples

```bash
uv run python train.py --help
uv run python scripts/play_pygame.py --help
```
