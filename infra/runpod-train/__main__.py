from __future__ import annotations

import sys
from pathlib import Path

import pulumi
import runpodinfra as runpod

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.runpod_infra import build_pod_env, build_train_start_command  # noqa: E402

cfg = pulumi.Config()

pod_name = cfg.get("podName") or "ataxx-trainer"
gpu_type_id = cfg.require("gpuTypeId")
image_name = cfg.get("imageName") or "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
cloud_type = cfg.get("cloudType") or "SECURE"
container_disk_in_gb = cfg.get_int("containerDiskInGb") or 30
volume_in_gb = cfg.get_int("volumeInGb") or 60
volume_mount_path = cfg.get("volumeMountPath") or "/workspace"
min_vcpu_count = cfg.get_int("minVcpuCount") or 8
min_memory_in_gb = cfg.get_int("minMemoryInGb") or 32
hf_repo_id = cfg.get("hfRepoId") or ""
hf_run_id = cfg.get("hfRunId") or "policy_spatial_v1"
hf_token = cfg.get_secret("hfToken")
repository = cfg.get("repository") or "dieg0code/ataxx-zero"
git_ref = cfg.get("gitRef") or "main"
train_args = cfg.get("trainArgs") or ""

docker_start_cmd = cfg.get("dockerStartCmd")
if docker_start_cmd is None or docker_start_cmd.strip() == "":
    docker_start_cmd = build_train_start_command(
        repository=repository,
        git_ref=git_ref,
        train_args=train_args,
        hf_repo_id=hf_repo_id,
        hf_run_id=hf_run_id,
    )

env = pulumi.Output.all(hf_token).apply(
    lambda values: build_pod_env(
        hf_token=str(values[0] or ""),
        hf_repo_id=hf_repo_id,
        hf_run_id=hf_run_id,
    )
)

# The pod is intentionally ephemeral: CI creates it, waits for the job, and destroys it.
trainer = runpod.Pod(
    "ataxx-trainer",
    name=pod_name,
    gpu_type_id=gpu_type_id,
    image_name=image_name,
    cloud_type=cloud_type,
    container_disk_in_gb=container_disk_in_gb,
    volume_in_gb=volume_in_gb,
    volume_mount_path=volume_mount_path,
    min_vcpu_count=min_vcpu_count,
    min_memory_in_gb=min_memory_in_gb,
    docker_start_cmd=docker_start_cmd,
    env=env,
)

pulumi.export("podId", trainer.id)
pulumi.export("podName", trainer.name)
