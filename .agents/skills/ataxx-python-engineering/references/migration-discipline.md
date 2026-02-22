# Migration Discipline (Alembic)

Use this checklist whenever DB models change.

## Rule

- Do not rely on `create_all` for production schema evolution.
- Use Alembic migrations for every structural model change.

## Mandatory checklist for model changes

1. Update SQLModel models.
2. Generate migration revision.
3. Review generated SQL (constraints, indexes, defaults, nullability).
4. Apply migration on local/dev DB.
5. Run tests.
6. Ensure downgrade path is valid when feasible.

## Practical workflow

```bash
# once alembic is configured
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
uv run pytest -q
```

## What requires a migration

- New table or dropped table.
- Added/removed column.
- Nullability changes.
- Type changes.
- Index/unique/foreign-key changes.
- Enum changes in DB-backed enum types.

## Review tips

- Confirm FK targets match actual table names (`user`, `game`, etc.).
- Confirm unique constraints are intentional and named.
- Confirm JSON columns keep expected type.
- Confirm server defaults match app defaults when needed.

## Integration-test expectation

- For API paths affected by schema changes, keep at least one integration test with real async DB (`sqlite+aiosqlite` in CI/local).
