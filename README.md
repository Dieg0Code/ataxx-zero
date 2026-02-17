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
- `--verbose`
- `--hf` enable Hugging Face upload
- `--hf-repo-id <org_or_user/repo>`

Examples:

Quick smoke run:

```bash
uv run python train.py --iterations 2 --episodes 8 --epochs 1 --sims 80 --batch-size 64 --save-every 1 --verbose
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
