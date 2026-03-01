from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from training.config_runtime import cfg_bool, cfg_str, log

if TYPE_CHECKING:
    from data.replay_buffer import ReplayBuffer
    from model.system import AtaxxZero


def cleanup_local_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    if keep_last_n < 1:
        keep_last_n = 1
    manual_files = sorted(checkpoint_dir.glob("manual_iter_*.ckpt"))
    if len(manual_files) <= keep_last_n:
        return
    for path in manual_files[:-keep_last_n]:
        path.unlink(missing_ok=True)
        log(f"Deleted old local checkpoint: {path.name}", verbose_only=True)


def cleanup_old_log_versions(log_dir: Path, run_name: str, keep_last_n: int) -> None:
    if keep_last_n < 1:
        keep_last_n = 1
    run_dir = log_dir / run_name
    if not run_dir.exists():
        return
    versions = sorted([p for p in run_dir.glob("version_*") if p.is_dir()])
    if len(versions) <= keep_last_n:
        return
    for old_dir in versions[:-keep_last_n]:
        for child in old_dir.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        for child_dir in sorted([p for p in old_dir.rglob("*") if p.is_dir()], reverse=True):
            child_dir.rmdir()
        old_dir.rmdir()
        log(f"Deleted old log version: {old_dir.name}", verbose_only=True)


class HuggingFaceCheckpointer:
    """Save and restore model/buffer state from Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str,
        token: str,
        run_id: str,
        local_dir: Path,
    ) -> None:
        hub_mod = __import__("huggingface_hub", fromlist=["HfApi"])
        api_cls = hub_mod.HfApi

        self.repo_id = repo_id
        self.token = token
        self.run_id = run_id
        self.local_dir = local_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.api = api_cls(token=token)
        self.api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    def _repo_path(self, filename: str) -> str:
        return f"runs/{self.run_id}/{filename}"

    def save_checkpoint_local(
        self,
        *,
        iteration: int,
        system: AtaxxZero,
        buffer: ReplayBuffer,
        config: dict[str, int | float | bool | str],
        stats: dict[str, float | int | str],
    ) -> tuple[Path, Path | None, Path]:
        model_name = f"model_iter_{iteration:03d}.pt"
        buffer_name = f"buffer_iter_{iteration:03d}.npz"
        metadata_name = f"metadata_iter_{iteration:03d}.json"

        model_path = self.local_dir / model_name
        torch.save(
            {
                "iteration": iteration,
                "state_dict": system.state_dict(),
                "hparams": dict(system.hparams),
            },
            model_path,
        )

        all_examples = buffer.get_all()
        buffer_path: Path | None = None
        if len(all_examples) > 0:
            observations = np.asarray([ex[0] for ex in all_examples], dtype=np.float32)
            policies = np.asarray([ex[1] for ex in all_examples], dtype=np.float32)
            values = np.asarray([ex[2] for ex in all_examples], dtype=np.float32)
            buffer_path = self.local_dir / buffer_name
            np.savez_compressed(
                buffer_path,
                observations=observations,
                policies=policies,
                values=values,
            )

        metadata: dict[str, Any] = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "buffer_size": len(buffer),
            "config": config,
            "stats": stats,
        }
        metadata_path = self.local_dir / metadata_name
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return model_path, buffer_path, metadata_path

    def upload_checkpoint_files(
        self,
        *,
        iteration: int,
        model_path: Path,
        buffer_path: Path | None,
        metadata_path: Path,
        keep_last_n: int,
    ) -> None:
        commit_message = f"Checkpoint iteration {iteration}"
        self.api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=self._repo_path(model_path.name),
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        if buffer_path is not None:
            self.api.upload_file(
                path_or_fileobj=str(buffer_path),
                path_in_repo=self._repo_path(buffer_path.name),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        self.api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo=self._repo_path(metadata_path.name),
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        self.cleanup_local_checkpoints(keep_last_n=keep_last_n)

    def load_latest_checkpoint(self, *, system: AtaxxZero, buffer: ReplayBuffer) -> int:
        hub_mod = __import__("huggingface_hub", fromlist=["hf_hub_download"])
        hf_hub_download = hub_mod.hf_hub_download

        files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model")
        run_prefix = self._repo_path("")
        model_files = [
            f
            for f in files
            if f.startswith(run_prefix)
            and Path(f).name.startswith("model_iter_")
            and (f.endswith(".pt") or f.endswith(".ckpt"))
        ]
        if len(model_files) == 0:
            return 0

        latest_iter = max(int(Path(name).stem.split("_")[2]) for name in model_files)
        model_name = f"model_iter_{latest_iter:03d}.pt"
        model_repo_path = self._repo_path(model_name)
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=model_repo_path,
            repo_type="model",
            token=self.token,
            local_dir=str(self.local_dir),
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Invalid checkpoint format in Hugging Face Hub.")
        state_dict_obj = checkpoint["state_dict"]
        if not isinstance(state_dict_obj, dict):
            raise ValueError("Checkpoint state_dict must be a dictionary.")
        try:
            system.load_state_dict(state_dict_obj)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint incompatible con architecture policy_head espacial; "
                "reentrena o usa carga parcial manual (strict=False)."
            ) from exc

        buffer_name = f"buffer_iter_{latest_iter:03d}.npz"
        buffer_repo_path = self._repo_path(buffer_name)
        try:
            buffer_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=buffer_repo_path,
                repo_type="model",
                token=self.token,
                local_dir=str(self.local_dir),
            )
            data = np.load(buffer_path)
            observations = data["observations"]
            policies = data["policies"]
            values = data["values"]
            examples = list(zip(observations, policies, values, strict=True))
            buffer.clear()
            buffer.save_game(examples)
        except (OSError, KeyError, ValueError):
            pass

        return latest_iter

    def cleanup_local_checkpoints(self, keep_last_n: int) -> None:
        model_files = sorted(self.local_dir.glob("model_iter_*.pt"))
        if len(model_files) <= keep_last_n:
            return
        old_files = model_files[:-keep_last_n]
        for model_file in old_files:
            iter_part = model_file.stem.split("_")[2]
            for pattern in (
                f"model_iter_{iter_part}.*",
                f"buffer_iter_{iter_part}.*",
                f"metadata_iter_{iter_part}.*",
            ):
                for path in self.local_dir.glob(pattern):
                    path.unlink(missing_ok=True)


def init_hf_checkpointer() -> HuggingFaceCheckpointer | None:
    if not cfg_bool("hf_enabled"):
        return None
    repo_id = cfg_str("hf_repo_id").strip()
    if repo_id == "":
        log("HF disabled: set CONFIG['hf_repo_id'] to enable checkpoint upload.")
        return None
    token_env = cfg_str("hf_token_env")
    token = str(os.environ.get(token_env, "")).strip()
    if token == "":
        log(f"HF disabled: missing token in env var '{token_env}'.")
        return None
    run_id = cfg_str("hf_run_id").strip()
    if run_id == "":
        log("HF disabled: set CONFIG['hf_run_id'] to namespace model checkpoints.")
        return None
    local_dir = Path(cfg_str("hf_local_dir"))
    return HuggingFaceCheckpointer(
        repo_id=repo_id,
        token=token,
        run_id=run_id,
        local_dir=local_dir,
    )


def should_save_iteration_checkpoint(iteration: int, total_iterations: int, save_every: int) -> bool:
    """Always persist the final iteration even when it is not divisible by save_every."""
    if iteration >= total_iterations:
        return True
    return iteration % save_every == 0


__all__ = [
    "HuggingFaceCheckpointer",
    "cleanup_local_checkpoints",
    "cleanup_old_log_versions",
    "init_hf_checkpointer",
    "should_save_iteration_checkpoint",
]
