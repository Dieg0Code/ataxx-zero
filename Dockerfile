# syntax=docker/dockerfile:1.7

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

# Install runtime dependencies only (api group + project deps).
RUN uv sync --frozen --no-dev --group api


FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MODEL_CHECKPOINT_PATH=/app/checkpoints/last.ckpt

WORKDIR /app

# Torch CPU wheels need libgomp at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]


# Optional target that bakes a local checkpoint into the image.
# Requires checkpoints/last.ckpt in build context.
FROM runtime AS runtime-with-model
COPY checkpoints/last.ckpt /app/checkpoints/last.ckpt
