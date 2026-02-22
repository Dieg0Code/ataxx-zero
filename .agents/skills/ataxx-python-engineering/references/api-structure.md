# API Structure

Use this target structure for FastAPI work in this repository.

```txt
src/api/
  app.py
  config/
    __init__.py
    settings.py
  db/
    __init__.py
    session.py
    enums.py
    models/
      __init__.py
      user.py
      model_version.py
      game.py
      game_move.py
    migrations/               # planned (alembic)
  deps/
    __init__.py
    inference.py
    gameplay.py
  modules/
    health/
      __init__.py
      router.py
    gameplay/
      __init__.py
      schemas.py
      repository.py
      service.py
      router.py
    identity/                 # planned
    ranking/                  # planned
    training_data/            # planned
```

## Layering rule

- Router -> Service -> Repository -> DB model.
- Keep business rules in services.
- Keep raw persistence in repositories.
- Keep request/response contracts in schemas.

## Dependency injection rule

- Keep per-feature deps in `src/api/deps/*`.
- Use dependency overrides in tests.
- Avoid global singletons for request-scoped state.

## Testing rule

- Unit tests for service/repository logic where useful.
- Integration tests with temporary async DB (`sqlite+aiosqlite`) for API flow.
- Keep API contract tests for status codes and payload shapes.
