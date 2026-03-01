# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS web-builder

WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# Build frontend so FastAPI can serve it from /web/static.
RUN npm run build -- --base=/web/static/


FROM python:3.10-slim AS py-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
# Runtime deps only (project + api group).
RUN uv sync --frozen --no-dev --group api


FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MODEL_CHECKPOINT_PATH=/app/checkpoints/last.ckpt

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 tini && \
    rm -rf /var/lib/apt/lists/*

COPY --from=py-builder /app/.venv /app/.venv
COPY src ./src
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY .env.example ./.env.example
# Replace legacy static bundle with current React build output.
COPY --from=web-builder /app/web/dist/ ./src/api/modules/web/static/

RUN mkdir -p /app/checkpoints

EXPOSE 8000

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]


# Optional target that bakes a local torch checkpoint into the image.
# Requires checkpoints/last.ckpt in build context.
FROM runtime AS runtime-with-model
COPY checkpoints/last.ckpt /app/checkpoints/last.ckpt

# Optional target for future ONNX runtime integration.
FROM runtime AS runtime-with-onnx
COPY checkpoints/last.onnx /app/checkpoints/last.onnx
