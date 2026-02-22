# Ataxx Zero

## Pygame (play / spectate)

Base:

```bash
uv run python scripts/play_pygame.py
```

Main flags:

- `--mode {play,spectate}`
- `--agent1 {human,random,heuristic,model}`
- `--agent2 {human,random,heuristic,model}`
- `--level1 {easy,normal,hard}` heuristic level for P1
- `--level2 {easy,normal,hard}` heuristic level for P2
- `--level {easy,normal,hard}` default level for both (fallback)
- `--ckpt <path>` model checkpoint
- `--sims <int>` MCTS simulations
- `--device {auto,cpu,cuda}`
- `--seed <int>` use `-1` for non-deterministic

Examples:

```bash
uv run python scripts/play_pygame.py --mode play --agent1 human --agent2 heuristic --level2 normal
uv run python scripts/play_pygame.py --mode spectate --agent1 heuristic --agent2 heuristic --level1 easy --level2 hard
uv run python scripts/play_pygame.py --mode spectate --agent1 model --agent2 heuristic --level2 hard --ckpt checkpoints/last.ckpt --sims 220
```

## Training

Main entrypoint is now:

```bash
uv run python train.py
```

`train_improved.py` is kept as compatibility wrapper.

## Dependency Profiles (uv)

Use dependency groups so each environment installs only what it needs.

- Base only:

```bash
uv sync
```

- API runtime (+dev tools):

```bash
uv sync --group api --group dev
```

- Training (+dev tools):

```bash
uv sync --group train --group dev
```

- Pygame UI (+dev tools):

```bash
uv sync --group ui --group dev
```

- ONNX export (+dev tools):

```bash
uv sync --group export --group dev
```

- Full environment (all groups):

```bash
uv sync --all-groups
```

Training flags:

- `--iterations <int>`
- `--episodes <int>` episodes per iteration
- `--sims <int>` MCTS simulations
- `--epochs <int>`
- `--batch-size <int>`
- `--lr <float>`
- `--weight-decay <float>`
- `--save-every <int>`
- `--seed <int>`
- `--checkpoint-dir <path>`
- `--log-dir <path>`
- `--onnx-path <path>`
- `--no-onnx` disable ONNX export at checkpoint time
- `--quiet` less console output (recommended for Kaggle)
- `--keep-local-ckpts <int>` local manual checkpoints to keep
- `--keep-log-versions <int>` TensorBoard versions to keep
- `--devices <int>` trainer devices (GPUs/accelerator processes)
- `--strategy <name>` Lightning strategy (`auto`, `ddp`, etc.)
- `--num-workers <int>` workers para DataLoader
- `--persistent-workers` mantiene workers vivos entre épocas (si `num-workers > 0`)
- `--no-persistent-workers` desactiva lo anterior
- `--strict-probs` valida que los porcentajes sumen 1.0 exacto
- `--no-eval` desactiva evaluacion periodica
- `--eval-every <int>` cada cuantas iteraciones evaluar
- `--eval-games <int>` numero de partidas de evaluacion
- `--eval-sims <int>` simulaciones MCTS durante evaluacion
- `--eval-heuristic-level {easy,normal,hard}` rival heuristico para evaluacion
- `--opp-self <float>` peso de oponente `self` (modelo vs sí mismo)
- `--opp-heuristic <float>` peso de oponente heurístico
- `--opp-random <float>` peso de oponente aleatorio
- `--opp-heuristic-level {easy,normal,hard}` nivel del heurístico en el pool
- `--opp-heu-easy <float>` peso de `easy` dentro del pool heurístico
- `--opp-heu-normal <float>` peso de `normal` dentro del pool heurístico
- `--opp-heu-hard <float>` peso de `hard` dentro del pool heurístico
- `--model-swap-prob <float>` probabilidad de cambiar de lado (P1/P2) por episodio
- `--verbose`
- `--hf` enable Hugging Face upload
- `--hf-repo-id <org_or_user/repo>`

Examples:

Quick smoke run:

```bash
uv run python train.py --iterations 2 --episodes 8 --epochs 1 --sims 80 --batch-size 64 --save-every 1 --verbose
```

Kaggle clean run (low logs + auto cleanup):

```bash
uv run python train.py --no-onnx --quiet --keep-local-ckpts 2 --keep-log-versions 1 --iterations 20 --episodes 50 --sims 300 --epochs 4 --batch-size 96 --lr 1e-3 --weight-decay 1e-4 --save-every 3
```

Kaggle 2x T4 (use both GPUs):

```bash
uv run python train.py --no-onnx --quiet --devices 2 --strategy ddp --keep-local-ckpts 2 --keep-log-versions 1 --hf --hf-repo-id your_user/ataxx-zero --iterations 40 --episodes 70 --sims 420 --epochs 5 --batch-size 96 --lr 9e-4 --weight-decay 1e-4 --save-every 3
```

Kaggle estable con `opponent pool` (recomendado):

```bash
uv run python train.py --no-onnx --quiet --devices 1 --strategy auto --keep-local-ckpts 2 --keep-log-versions 1 --hf --hf-repo-id your_user/ataxx-zero --iterations 40 --episodes 70 --sims 420 --epochs 5 --batch-size 96 --lr 9e-4 --weight-decay 1e-4 --save-every 3 --opp-self 0.80 --opp-heuristic 0.15 --opp-random 0.05 --opp-heu-easy 0.20 --opp-heu-normal 0.50 --opp-heu-hard 0.30 --model-swap-prob 0.5
```

Kaggle estable + evaluacion automatica + best checkpoint:

```bash
uv run python train.py --no-onnx --quiet --devices 1 --strategy auto --num-workers 3 --persistent-workers --keep-local-ckpts 2 --keep-log-versions 1 --hf --hf-repo-id your_user/ataxx-zero --iterations 40 --episodes 70 --sims 420 --epochs 5 --batch-size 96 --lr 9e-4 --weight-decay 1e-4 --save-every 3 --strict-probs --eval-every 3 --eval-games 12 --eval-sims 220 --eval-heuristic-level hard --opp-self 0.85 --opp-heuristic 0.12 --opp-random 0.03 --opp-heu-easy 0.05 --opp-heu-normal 0.20 --opp-heu-hard 0.75 --model-swap-prob 0.5
```

If your environment is missing ONNX tooling, use:

```bash
uv run python train.py --no-onnx ...
```

Standard local run:

```bash
uv run python train.py --iterations 20 --episodes 50 --epochs 5 --sims 400 --batch-size 128 --lr 1e-3 --weight-decay 1e-4
```

With Hugging Face checkpoint upload:

```bash
# set token first (PowerShell)
$env:HF_TOKEN="your_token_here"
uv run python train.py --hf --hf-repo-id your_user/ataxx-zero --save-every 5
```

## Training Profiles (Colab / Kaggle)

### 1) Pro mode (max strength, slower)

Use this when you have a good GPU session and want best quality per run.

```bash
uv run python train.py \
  --iterations 40 \
  --episodes 120 \
  --sims 600 \
  --epochs 8 \
  --batch-size 128 \
  --lr 8e-4 \
  --weight-decay 1e-4 \
  --save-every 5 \
  --verbose \
  --hf --hf-repo-id your_user/ataxx-zero
```

### 2) Decent but fast (recommended default)

Good quality/speed balance for regular experimentation.

```bash
uv run python train.py \
  --iterations 20 \
  --episodes 50 \
  --sims 300 \
  --epochs 4 \
  --batch-size 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --save-every 5
```

### 3) Fast debug mode (sanity check)

Use this to verify pipeline, logging, checkpoints, and no-NaN behavior quickly.

```bash
uv run python train.py \
  --iterations 3 \
  --episodes 10 \
  --sims 80 \
  --epochs 1 \
  --batch-size 64 \
  --save-every 1 \
  --verbose
```

### 4) Strong self-play, lighter training

When you want stronger targets (MCTS) but cheaper gradient updates.

```bash
uv run python train.py \
  --iterations 18 \
  --episodes 70 \
  --sims 500 \
  --epochs 3 \
  --batch-size 96 \
  --lr 9e-4 \
  --save-every 3
```

### 5) Quick resume from cloud checkpoints

If you already have HF checkpoints and only want to continue training.

```bash
$env:HF_TOKEN="your_token_here"
uv run python train.py \
  --iterations 30 \
  --episodes 40 \
  --sims 250 \
  --epochs 3 \
  --hf --hf-repo-id your_user/ataxx-zero
```

Notes for Colab/Kaggle:

- If runtime time is limited, prioritize lowering `--episodes` first, then `--iterations`.
- `--sims` has the biggest impact on self-play quality and runtime.
- If you hit memory limits, lower `--batch-size` to `96` or `64`.
- Keep `--save-every` small (`3` to `5`) when using temporary sessions.

## GPU Guide (Colab / Kaggle)

### T4 (entry-level, very common)

Best choice: decent-but-fast profile.

```bash
uv run python train.py \
  --iterations 16 \
  --episodes 35 \
  --sims 220 \
  --epochs 3 \
  --batch-size 96 \
  --lr 1e-3 \
  --save-every 4
```

### L4 (strong balance)

Best choice: strong self-play with moderate epochs.

```bash
uv run python train.py \
  --iterations 24 \
  --episodes 60 \
  --sims 380 \
  --epochs 4 \
  --batch-size 128 \
  --lr 9e-4 \
  --save-every 4
```

### A100 (high-end)

Best choice: pro mode.

```bash
uv run python train.py \
  --iterations 45 \
  --episodes 130 \
  --sims 700 \
  --epochs 8 \
  --batch-size 192 \
  --lr 8e-4 \
  --weight-decay 1e-4 \
  --save-every 5 \
  --hf --hf-repo-id your_user/ataxx-zero
```

Quick rule:

- `T4`: prioritize shorter runs and checkpoint often.
- `L4`: use as default if available.
- `A100`: maximize self-play quality (`--sims`, `--episodes`) and larger batch.

## API (FastAPI)

Install API environment:

```bash
uv sync --group api --group dev
```

Run server:

```bash
uv run uvicorn api.app:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

Web UI (browser):

```bash
http://127.0.0.1:8000/web
```

The web UI is a first playable version (Human P1 vs AI P2) and calls:

- `POST /api/v1/gameplay/move` for AI decisions.

## Web Frontend (React + Vite + TS + Tailwind)

Scaffold location:

```txt
web/
```

Install dependencies:

```bash
cd web
npm install
```

Run in development:

```bash
npm run dev
```

Build production assets:

```bash
npm run build
```

Design direction:

- Brand: `underbyteLabs - ataxx-zero`
- Mobile-first layout
- Public ranking
- Multi-skin theme system: `terminal-neo`, `amber-crt`, `oxide-red`

### Docker (API + DB)

Build API image (multi-stage, runtime target):

```bash
docker build -t ataxx-api:latest --target runtime .
```

Run API + Postgres with compose:

```bash
docker compose up --build
```

API will be available at:

```bash
http://127.0.0.1:8000
```

Model checkpoint handling in Docker:

- Default compose mounts local `./checkpoints` into container as read-only.
- API expects checkpoint at `MODEL_CHECKPOINT_PATH` (default `/app/checkpoints/last.ckpt`).
- If checkpoint is missing, inference endpoints return `503`.

Optional: bake checkpoint into image:

```bash
docker build -t ataxx-api:with-model --target runtime-with-model .
```

Then run without checkpoint volume (or keep it mounted).

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Readiness check (includes DB connectivity):

```bash
curl http://127.0.0.1:8000/health/ready
```

CORS configuration (via `.env`):

```dotenv
APP_CORS_ORIGINS=["http://localhost:5173"]
APP_CORS_ALLOW_CREDENTIALS=true
APP_CORS_ALLOW_METHODS=["*"]
APP_CORS_ALLOW_HEADERS=["*"]
```

Observability configuration (via `.env`):

```dotenv
APP_LOG_LEVEL="INFO"
APP_LOG_JSON=true
APP_LOG_REQUESTS=true
```

When `APP_LOG_REQUESTS=true`, each request logs method/path/status/duration/request_id.

### Database Migrations (Alembic)

Alembic is configured in this repo for SQLModel metadata under `src/api/db/models`.

Install dependencies:

```bash
uv sync --group api --group dev
```

Check migration status:

```bash
uv run alembic current
uv run alembic heads
```

Apply all migrations:

```bash
uv run alembic upgrade head
```

Create a new migration after changing models:

```bash
uv run alembic revision --autogenerate -m "describe change"
```

Rollback one migration:

```bash
uv run alembic downgrade -1
```

PowerShell shortcut script:

```powershell
.\scripts\db.ps1 up
.\scripts\db.ps1 down
.\scripts\db.ps1 new "add user profile fields"
.\scripts\db.ps1 current
.\scripts\db.ps1 heads
```

Notes:

- Alembic reads DB connection from `.env` through `api.config.settings`.
- Use migrations (`alembic upgrade`) for shared/prod DB.
- `init_db()` remains useful for isolated tests/local ephemeral DB only.

### API Pagination (offset + limit)

List endpoints now use a common paginated shape:

```json
{
  "items": [],
  "total": 0,
  "limit": 20,
  "offset": 0,
  "has_more": false
}
```

Supported list endpoints:

- `GET /api/v1/gameplay/games?limit=20&offset=0`
- `GET /api/v1/training/samples?limit=100&offset=0`
- `GET /api/v1/model-versions?limit=50&offset=0`
- `GET /api/v1/ranking/leaderboard/{season_id}?limit=100&offset=0`
- `GET /api/v1/identity/users?limit=50&offset=0` (admin)

Examples:

```bash
curl "http://127.0.0.1:8000/api/v1/gameplay/games?limit=10&offset=0" -H "Authorization: Bearer <ACCESS_TOKEN>"
curl "http://127.0.0.1:8000/api/v1/training/samples?limit=25&offset=25&split=train"
curl "http://127.0.0.1:8000/api/v1/model-versions?limit=10&offset=0"
curl "http://127.0.0.1:8000/api/v1/ranking/leaderboard/<SEASON_ID>?limit=20&offset=0"
curl "http://127.0.0.1:8000/api/v1/identity/users?limit=10&offset=0" -H "Authorization: Bearer <ADMIN_ACCESS_TOKEN>"
```

Pagination behavior:

- `limit` is clamped per endpoint for safety.
- `offset` starts at `0`.
- `has_more=true` means you can request the next page with `offset + limit`.

### Auth Flow (end-to-end)

Register:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"diego\",\"email\":\"diego@example.com\",\"password\":\"supersecret123\"}"
```

Login (returns `access_token` + `refresh_token`):

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username_or_email\":\"diego\",\"password\":\"supersecret123\"}"
```

Sample login response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

Get current user (`/me`) with access token:

```bash
curl "http://127.0.0.1:8000/api/v1/auth/me" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

Refresh tokens:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\":\"<REFRESH_TOKEN>\"}"
```

Logout (revoke refresh token):

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/logout" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\":\"<REFRESH_TOKEN>\"}"
```

Notes:

- Send `Authorization: Bearer <ACCESS_TOKEN>` to protected endpoints.
- After `refresh`, prefer replacing both stored tokens.
- `logout` revokes refresh token; current access token remains valid until expiry.
- `login` and `refresh` are rate-limited (returns `429` + `Retry-After` header).

Auth error examples (standard error envelope):

401 Unauthorized (missing/invalid token):

```json
{
  "error_code": "unauthorized",
  "message": "Not authenticated",
  "detail": "Not authenticated",
  "request_id": "req-123"
}
```

403 Forbidden (insufficient permissions):

```json
{
  "error_code": "forbidden",
  "message": "Admin privileges required.",
  "detail": "Admin privileges required.",
  "request_id": "req-456"
}
```

422 Validation Error (invalid payload):

```json
{
  "error_code": "validation_error",
  "message": "Validation failed",
  "detail": "Validation failed",
  "request_id": "req-789",
  "details": [
    {
      "type": "missing",
      "loc": ["body", "password"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

Gameplay/Matches error examples:

400 Bad Request (invalid board / illegal move):

```json
{
  "error_code": "bad_request",
  "message": "Illegal move for current board state.",
  "detail": "Illegal move for current board state.",
  "request_id": "req-101"
}
```

403 Forbidden (not participant / not your turn):

```json
{
  "error_code": "forbidden",
  "message": "It is not your turn.",
  "detail": "It is not your turn.",
  "request_id": "req-102"
}
```

404 Not Found (game/match/sample/version missing):

```json
{
  "error_code": "not_found",
  "message": "Game not found: <uuid>",
  "detail": "Game not found: <uuid>",
  "request_id": "req-103"
}
```

### Training Samples API

Create a game (needed for sample FK):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gameplay/games -H "Content-Type: application/json" -d "{}"
```

Create one training sample:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/training/samples -H "Content-Type: application/json" -d "{\"game_id\":\"<GAME_ID>\",\"ply\":0,\"player_side\":\"p1\",\"observation\":{\"grid\":[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],\"current_player\":1},\"policy_target\":{\"10\":1.0},\"value_target\":1.0,\"sample_weight\":1.0,\"split\":\"train\",\"source\":\"self_play\"}"
```

List samples:

```bash
curl "http://127.0.0.1:8000/api/v1/training/samples?limit=50&split=train"
```

Samples stats:

```bash
curl "http://127.0.0.1:8000/api/v1/training/samples/stats?split=train"
```

Export samples as NDJSON:

```bash
curl "http://127.0.0.1:8000/api/v1/training/samples/export.ndjson?split=train&limit=500" -o training_samples.ndjson
```

Export samples as NPZ:

```bash
curl "http://127.0.0.1:8000/api/v1/training/samples/export.npz?split=train&limit=500" -o training_samples.npz
```

Ingest samples from a finished game:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/training/samples/ingest-game/<GAME_ID>?split=train&source=self_play&overwrite=true"
```
