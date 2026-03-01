# Entrenamiento RunPod con Pulumi

Este stack crea un pod efimero en RunPod para entrenar y exporta `podId`.

## Workflows recomendados

Usa:
- `.github/workflows/train-runpod-start.yml` para crear/iniciar el pod.
- `.github/workflows/train-runpod-reconcile.yml` para revisar estado y destruir cuando termine.

Este stack es solo para entrenamiento en RunPod. El deploy de app web/backend en Railway
se maneja con autodeploy desde GitHub + `CI API` / `CI Web`.

Secrets requeridos en el repositorio:
- `RUNPOD_API_TOKEN`
- `HF_TOKEN`
- `PULUMI_ACCESS_TOKEN`

Variable recomendada para el reconcile programado:
- `RUNPOD_TRAIN_STACK` (example: `dieg0code/train`)

Iniciar entrenamiento desde CLI:

```bash
gh workflow run train-runpod-start.yml \
  --ref main \
  -f stack=dieg0code/train \
  -f pod_name=ataxx-zero-train \
  -f gpu_type_id="NVIDIA GeForce RTX 4090" \
  -f cloud_type=SECURE \
  -f image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  -f repository=dieg0code/ataxx-zero \
  -f git_ref=main \
  -f hf_repo_id=dieg0code/ataxx-zero \
  -f hf_run_id=policy_spatial_v1 \
  -f train_args="--no-onnx --quiet --devices 1 --strategy auto --num-workers 4 --keep-local-ckpts 2 --keep-log-versions 1 --hf --iterations 40 --episodes 70 --sims 600 --epochs 5 --batch-size 512 --lr 9e-4 --weight-decay 1e-4 --save-every 3 --opp-self 0.45 --opp-heuristic 0.50 --opp-random 0.05 --opp-heu-easy 0.00 --opp-heu-normal 0.25 --opp-heu-hard 0.75 --model-swap-prob 0.5 --selfplay-workers 8 --monitor-log-every 3"
```

Reconcile manual (opcional, ya existe cron cada 30 min):

```bash
gh workflow run train-runpod-reconcile.yml \
  --ref main \
  -f stack=dieg0code/train
```

Comportamiento esperado:

- El workflow `start` no espera a que termine el entrenamiento.
- El pod entrena en RunPod de forma independiente.
- `reconcile` destruye pods terminales y elimina el stack para cortar cobro.
