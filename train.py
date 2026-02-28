from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytorch_lightning as pl
import torch
from huggingface_hub import HfApi, hf_hub_download
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from data.replay_buffer import ReplayBuffer, TrainingExample
    from engine.mcts import MCTS
    from game.board import AtaxxBoard
    from model.system import AtaxxZero


_WORKER_MCTS: object | None = None


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


CONFIG: dict[str, int | float | bool | str] = {
    "iterations": 20,
    "episodes_per_iter": 50,
    "mcts_sims": 400,
    "c_puct": 1.5,
    "temp_threshold": 15,
    "add_noise": True,
    "seed": 42,
    "verbose_logs": False,
    "episode_log_every": 25,
    "epochs": 5,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "buffer_size": 50_000,
    "val_split": 0.1,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "log_dir": "logs",
    "checkpoint_dir": "checkpoints",
    "save_every": 5,
    "keep_last_n_local_checkpoints": 3,
    "keep_last_n_log_versions": 2,
    "keep_last_n_hf_checkpoints": 3,
    "onnx_path": "ataxx_model.onnx",
    "export_onnx": True,
    "hf_enabled": False,
    "hf_repo_id": "",
    "hf_token_env": "HF_TOKEN",
    "hf_local_dir": "hf_checkpoints",
    "show_progress_bar": True,
    "trainer_log_every_n_steps": 10,
    "num_workers": 0,
    "persistent_workers": True,
    "strict_probs": False,
    "trainer_devices": 1,
    "trainer_strategy": "auto",
    "trainer_precision": "16-mixed",
    "trainer_benchmark": True,
    "mcts_use_amp": True,
    "opponent_self_prob": 0.8,
    "opponent_heuristic_prob": 0.15,
    "opponent_random_prob": 0.05,
    "opponent_heuristic_level": "normal",
    "opponent_heuristic_easy_prob": 0.2,
    "opponent_heuristic_normal_prob": 0.5,
    "opponent_heuristic_hard_prob": 0.3,
    "model_side_swap_prob": 0.5,
    "eval_enabled": True,
    "eval_every": 3,
    "eval_games": 12,
    "eval_sims": 220,
    "eval_heuristic_level": "hard",
    "selfplay_workers": 1,
    "quiet_mode": False,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ataxx Zero.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--keep-local-ckpts", type=int, default=None)
    parser.add_argument("--keep-log-versions", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument(
        "--precision",
        choices=["16-mixed", "bf16-mixed", "32-true"],
        default=None,
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--strict-probs", action="store_true")
    parser.add_argument("--no-mcts-amp", action="store_true")
    parser.add_argument("--opp-self", type=float, default=None)
    parser.add_argument("--opp-heuristic", type=float, default=None)
    parser.add_argument("--opp-random", type=float, default=None)
    parser.add_argument(
        "--opp-heuristic-level",
        choices=["easy", "normal", "hard"],
        default=None,
    )
    parser.add_argument("--opp-heu-easy", type=float, default=None)
    parser.add_argument("--opp-heu-normal", type=float, default=None)
    parser.add_argument("--opp-heu-hard", type=float, default=None)
    parser.add_argument("--model-swap-prob", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-sims", type=int, default=None)
    parser.add_argument("--selfplay-workers", type=int, default=None)
    parser.add_argument(
        "--eval-heuristic-level",
        choices=["easy", "normal", "hard"],
        default=None,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.persistent_workers and args.no_persistent_workers:
        raise ValueError("Use only one of --persistent-workers or --no-persistent-workers.")
    if args.iterations is not None:
        CONFIG["iterations"] = args.iterations
    if args.episodes is not None:
        CONFIG["episodes_per_iter"] = args.episodes
    if args.sims is not None:
        CONFIG["mcts_sims"] = args.sims
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
    if args.batch_size is not None:
        CONFIG["batch_size"] = args.batch_size
    if args.lr is not None:
        CONFIG["learning_rate"] = args.lr
    if args.weight_decay is not None:
        CONFIG["weight_decay"] = args.weight_decay
    if args.save_every is not None:
        CONFIG["save_every"] = args.save_every
    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.checkpoint_dir is not None:
        CONFIG["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        CONFIG["log_dir"] = args.log_dir
    if args.onnx_path is not None:
        CONFIG["onnx_path"] = args.onnx_path
    if args.no_onnx:
        CONFIG["export_onnx"] = False
    if args.keep_local_ckpts is not None:
        CONFIG["keep_last_n_local_checkpoints"] = args.keep_local_ckpts
    if args.keep_log_versions is not None:
        CONFIG["keep_last_n_log_versions"] = args.keep_log_versions
    if args.devices is not None:
        CONFIG["trainer_devices"] = max(1, args.devices)
    if args.strategy is not None:
        CONFIG["trainer_strategy"] = args.strategy
    if args.precision is not None:
        CONFIG["trainer_precision"] = args.precision
    if args.num_workers is not None:
        CONFIG["num_workers"] = max(0, args.num_workers)
    if args.persistent_workers:
        CONFIG["persistent_workers"] = True
    if args.no_persistent_workers:
        CONFIG["persistent_workers"] = False
    if args.strict_probs:
        CONFIG["strict_probs"] = True
    if args.no_mcts_amp:
        CONFIG["mcts_use_amp"] = False
    if args.opp_self is not None:
        CONFIG["opponent_self_prob"] = max(0.0, args.opp_self)
    if args.opp_heuristic is not None:
        CONFIG["opponent_heuristic_prob"] = max(0.0, args.opp_heuristic)
    if args.opp_random is not None:
        CONFIG["opponent_random_prob"] = max(0.0, args.opp_random)
    if args.opp_heuristic_level is not None:
        CONFIG["opponent_heuristic_level"] = args.opp_heuristic_level
    if args.opp_heu_easy is not None:
        CONFIG["opponent_heuristic_easy_prob"] = max(0.0, args.opp_heu_easy)
    if args.opp_heu_normal is not None:
        CONFIG["opponent_heuristic_normal_prob"] = max(0.0, args.opp_heu_normal)
    if args.opp_heu_hard is not None:
        CONFIG["opponent_heuristic_hard_prob"] = max(0.0, args.opp_heu_hard)
    if args.model_swap_prob is not None:
        CONFIG["model_side_swap_prob"] = min(max(args.model_swap_prob, 0.0), 1.0)
    if args.no_eval:
        CONFIG["eval_enabled"] = False
    if args.eval_every is not None:
        CONFIG["eval_every"] = max(1, args.eval_every)
    if args.eval_games is not None:
        CONFIG["eval_games"] = max(2, args.eval_games)
    if args.eval_sims is not None:
        CONFIG["eval_sims"] = max(8, args.eval_sims)
    if args.selfplay_workers is not None:
        CONFIG["selfplay_workers"] = max(1, args.selfplay_workers)
    if args.eval_heuristic_level is not None:
        CONFIG["eval_heuristic_level"] = args.eval_heuristic_level
    if args.quiet:
        CONFIG["show_progress_bar"] = False
        CONFIG["trainer_log_every_n_steps"] = 100
        CONFIG["episode_log_every"] = 0
        CONFIG["quiet_mode"] = True
    if args.verbose:
        CONFIG["verbose_logs"] = True
    if args.hf:
        CONFIG["hf_enabled"] = True
    if args.hf_repo_id is not None:
        CONFIG["hf_repo_id"] = args.hf_repo_id


def _cfg_int(key: str) -> int:
    return int(CONFIG[key])


def _cfg_float(key: str) -> float:
    return float(CONFIG[key])


def _cfg_bool(key: str) -> bool:
    return bool(CONFIG[key])


def _cfg_str(key: str) -> str:
    return str(CONFIG[key])


def _is_quiet() -> bool:
    return _cfg_bool("quiet_mode")


def _validate_config() -> None:
    int_positive_keys = (
        "iterations",
        "episodes_per_iter",
        "mcts_sims",
        "epochs",
        "batch_size",
        "save_every",
        "eval_every",
        "eval_games",
        "eval_sims",
        "selfplay_workers",
    )
    for key in int_positive_keys:
        if _cfg_int(key) <= 0:
            raise ValueError(f"CONFIG['{key}'] must be > 0, got {_cfg_int(key)}.")
    if _cfg_int("num_workers") < 0:
        raise ValueError("CONFIG['num_workers'] must be >= 0.")

    opp_sum = (
        _cfg_float("opponent_self_prob")
        + _cfg_float("opponent_heuristic_prob")
        + _cfg_float("opponent_random_prob")
    )
    heu_sum = (
        _cfg_float("opponent_heuristic_easy_prob")
        + _cfg_float("opponent_heuristic_normal_prob")
        + _cfg_float("opponent_heuristic_hard_prob")
    )
    if _cfg_bool("strict_probs"):
        if not np.isclose(opp_sum, 1.0, atol=1e-6):
            raise ValueError(
                "Opponent probs must sum to 1.0 when --strict-probs is enabled "
                f"(got {opp_sum:.6f}).",
            )
        if not np.isclose(heu_sum, 1.0, atol=1e-6):
            raise ValueError(
                "Heuristic level probs must sum to 1.0 when --strict-probs is enabled "
                f"(got {heu_sum:.6f}).",
            )
    if not _cfg_bool("strict_probs") and not np.isclose(opp_sum, 1.0, atol=1e-6):
        _log(
            f"Opponent probs sum to {opp_sum:.6f}; they will be normalized automatically.",
            verbose_only=True,
        )
    if not _cfg_bool("strict_probs") and not np.isclose(heu_sum, 1.0, atol=1e-6):
        _log(
            f"Heuristic level probs sum to {heu_sum:.6f}; they will be normalized automatically.",
            verbose_only=True,
        )


def _log(message: str, verbose_only: bool = False) -> None:
    if verbose_only and not _cfg_bool("verbose_logs"):
        return
    print(message)


def _resolve_trainer_hw() -> tuple[str, int, str]:
    requested_devices = _cfg_int("trainer_devices")
    strategy = _cfg_str("trainer_strategy")
    if torch.cuda.is_available():
        available = max(1, torch.cuda.device_count())
        devices = min(requested_devices, available)
        if requested_devices > available:
            _log(
                f"Requested {requested_devices} GPU(s), but only {available} available. Using {devices}.",
            )
        if strategy == "auto" and devices > 1:
            strategy = "ddp"
        return "gpu", devices, strategy
    if requested_devices > 1:
        _log("CUDA not available, forcing devices=1 on CPU.")
    return "cpu", 1, strategy


def _is_ddp_rendezvous_timeout(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        ("DistStoreError" in msg or "init_process_group" in msg)
        and ("clients joined" in msg or "Timed out" in msg)
    )


def _resolve_trainer_precision(accelerator: str) -> str:
    configured = _cfg_str("trainer_precision")
    if accelerator == "gpu":
        return configured
    return "32-true"


def _build_trainer(
    *,
    epochs: int,
    accelerator: str,
    devices: int,
    strategy: str,
    precision: str,
    benchmark: bool,
    checkpoint_callback: ModelCheckpoint,
    lr_monitor: LearningRateMonitor,
    logger: TensorBoardLogger,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        benchmark=benchmark,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=_cfg_bool("show_progress_bar"),
        log_every_n_steps=_cfg_int("trainer_log_every_n_steps"),
        gradient_clip_val=1.0,
    )


def _cleanup_local_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    if keep_last_n < 1:
        keep_last_n = 1
    manual_files = sorted(checkpoint_dir.glob("manual_iter_*.ckpt"))
    if len(manual_files) <= keep_last_n:
        return
    for path in manual_files[:-keep_last_n]:
        path.unlink(missing_ok=True)
        _log(f"Deleted old local checkpoint: {path.name}", verbose_only=True)


def _cleanup_old_log_versions(log_dir: Path, run_name: str, keep_last_n: int) -> None:
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
        _log(f"Deleted old log version: {old_dir.name}", verbose_only=True)


class HuggingFaceCheckpointer:
    """Save and restore model/buffer state from Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str,
        token: str,
        local_dir: Path,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    def save_checkpoint_local(
        self,
        *,
        iteration: int,
        system: AtaxxZero,
        buffer: ReplayBuffer,
        config: dict[str, int | float | bool | str],
        stats: dict[str, float | int],
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
            path_in_repo=model_path.name,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        if buffer_path is not None:
            self.api.upload_file(
                path_or_fileobj=str(buffer_path),
                path_in_repo=buffer_path.name,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        self.api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo=metadata_path.name,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        self.cleanup_local_checkpoints(keep_last_n=keep_last_n)

    def load_latest_checkpoint(self, *, system: AtaxxZero, buffer: ReplayBuffer) -> int:
        files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model")
        model_files = [
            f
            for f in files
            if f.startswith("model_iter_") and (f.endswith(".pt") or f.endswith(".ckpt"))
        ]
        if len(model_files) == 0:
            return 0

        latest_iter = max(int(Path(name).stem.split("_")[2]) for name in model_files)
        model_name = f"model_iter_{latest_iter:03d}.pt"
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=model_name,
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
        system.load_state_dict(state_dict_obj)

        buffer_name = f"buffer_iter_{latest_iter:03d}.npz"
        try:
            buffer_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=buffer_name,
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


def _init_hf_checkpointer() -> HuggingFaceCheckpointer | None:
    if not _cfg_bool("hf_enabled"):
        return None
    repo_id = _cfg_str("hf_repo_id").strip()
    if repo_id == "":
        _log("HF disabled: set CONFIG['hf_repo_id'] to enable checkpoint upload.")
        return None
    token_env = _cfg_str("hf_token_env")
    token = str(os.environ.get(token_env, "")).strip()
    if token == "":
        _log(f"HF disabled: missing token in env var '{token_env}'.")
        return None
    local_dir = Path(_cfg_str("hf_local_dir"))
    return HuggingFaceCheckpointer(repo_id=repo_id, token=token, local_dir=local_dir)


def _compute_action_probs(
    board: AtaxxBoard,
    mcts: MCTS,
    root: object | None,
    add_noise: bool,
    temperature: float,
) -> tuple[np.ndarray, object | None]:
    from game.actions import ACTION_SPACE

    probs, updated_root = mcts.run_with_root(
        board=board,
        root=root,
        add_dirichlet_noise=add_noise,
        temperature=temperature,
    )
    total_prob = float(np.sum(probs))
    if total_prob > 0.0:
        return probs, updated_root

    valid_moves = board.get_valid_moves()
    fallback = ACTION_SPACE.mask_from_moves(
        valid_moves,
        include_pass=(len(valid_moves) == 0),
    )
    return fallback / float(np.sum(fallback)), updated_root


def _select_action_idx(
    probs: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature > 0.0:
        return int(rng.choice(len(probs), p=probs))
    return int(np.argmax(probs))


def _score_move_for_player(board: AtaxxBoard, move: tuple[int, int, int, int]) -> float:
    r1, c1, r2, c2 = move
    board_size = int(board.grid.shape[0])
    jump = 1 if max(abs(r2 - r1), abs(c2 - c1)) == 2 else 0
    r_min = max(0, r2 - 1)
    r_max = min(board_size, r2 + 2)
    c_min = max(0, c2 - 1)
    c_max = min(board_size, c2 + 2)
    neighborhood = board.grid[r_min:r_max, c_min:c_max]
    converted = float(np.sum(neighborhood == -board.current_player))
    center = float(board_size - 1) / 2.0
    center_bonus = 0.35 * ((board_size - 1) - abs(r2 - center) - abs(c2 - center))
    return converted + center_bonus - 0.55 * float(jump)


def _heuristic_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
    level: str,
) -> tuple[int, int, int, int] | None:
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return None
    if level == "easy":
        return moves[int(rng.integers(0, len(moves)))]

    if level == "hard":
        top_k = max(1, min(3, len(moves)))
    else:
        top_k = max(1, min(5, len(moves)))
    scored = heapq.nlargest(
        top_k,
        ((move, _score_move_for_player(board, move)) for move in moves),
        key=lambda item: item[1],
    )
    weights = np.linspace(1.0, 0.35, top_k, dtype=np.float64)
    weights = weights / np.sum(weights)
    pick = int(rng.choice(top_k, p=weights))
    return scored[pick][0]


def _random_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return None
    return moves[int(rng.integers(0, len(moves)))]


def _resolve_opponent_mix() -> tuple[np.ndarray, tuple[str, str, str]]:
    labels = ("self", "heuristic", "random")
    raw = np.array(
        [
            _cfg_float("opponent_self_prob"),
            _cfg_float("opponent_heuristic_prob"),
            _cfg_float("opponent_random_prob"),
        ],
        dtype=np.float64,
    )
    raw = np.maximum(raw, 0.0)
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), labels
    return raw / total, labels


def _sample_opponent_type(rng: np.random.Generator) -> str:
    probs, labels = _resolve_opponent_mix()
    idx = int(rng.choice(len(labels), p=probs))
    return labels[idx]


def _resolve_heuristic_level_mix() -> tuple[np.ndarray, tuple[str, str, str]]:
    levels = ("easy", "normal", "hard")
    raw = np.array(
        [
            _cfg_float("opponent_heuristic_easy_prob"),
            _cfg_float("opponent_heuristic_normal_prob"),
            _cfg_float("opponent_heuristic_hard_prob"),
        ],
        dtype=np.float64,
    )
    raw = np.maximum(raw, 0.0)
    total = float(np.sum(raw))
    if total <= 0.0:
        fallback = _cfg_str("opponent_heuristic_level")
        if fallback == "easy":
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), levels
        if fallback == "hard":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64), levels
        return np.array([0.0, 1.0, 0.0], dtype=np.float64), levels
    return raw / total, levels


def _sample_heuristic_level(rng: np.random.Generator) -> str:
    probs, levels = _resolve_heuristic_level_mix()
    idx = int(rng.choice(len(levels), p=probs))
    return levels[idx]


def _play_episode(
    mcts: MCTS,
    add_noise: bool,
    temp_threshold: int,
    rng: np.random.Generator,
    opponent_type: str,
    opponent_heuristic_level: str,
    model_player: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int, int]:
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard

    board = AtaxxBoard()
    root = None
    game_history: list[tuple[np.ndarray, np.ndarray, int]] = []
    turn_idx = 0

    while not board.is_game_over():
        turn_idx += 1
        is_model_turn = board.current_player == model_player
        if is_model_turn or opponent_type == "self":
            temperature = 1.0 if turn_idx <= temp_threshold else 0.0
            probs, root = _compute_action_probs(
                board=board,
                mcts=mcts,
                root=root,
                add_noise=(add_noise and is_model_turn),
                temperature=temperature,
            )
            game_history.append(
                (
                    board.get_observation(),
                    probs,
                    board.current_player,
                )
            )
            action_idx = _select_action_idx(
                probs=probs,
                temperature=temperature,
                rng=rng,
            )
            board.step(ACTION_SPACE.decode(action_idx))
            root = mcts.advance_root(root, action_idx)
            continue

        if opponent_type == "heuristic":
            move = _heuristic_move(board, rng, opponent_heuristic_level)
            board.step(move)
            root = mcts.advance_root(root, ACTION_SPACE.encode(move))
            continue

        move = _random_move(board, rng)
        board.step(move)
        root = mcts.advance_root(root, ACTION_SPACE.encode(move))

    return game_history, board.get_result(), turn_idx


def _init_selfplay_process_worker(
    model_state_dict: dict[str, torch.Tensor],
    model_cfg: dict[str, int | float],
    c_puct: float,
    sims: int,
) -> None:
    global _WORKER_MCTS
    _ensure_src_on_path()
    from engine.mcts import MCTS
    from model.transformer import AtaxxTransformerNet

    model = AtaxxTransformerNet(
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    _WORKER_MCTS = MCTS(
        model=model,
        c_puct=c_puct,
        n_simulations=sims,
        device="cpu",
        use_amp=False,
    )


def _run_episode_in_process_worker(
    payload: tuple[int, str, str, int, bool, int],
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int, int]:
    global _WORKER_MCTS
    if _WORKER_MCTS is None:
        raise RuntimeError("Worker MCTS is not initialized.")
    episode_seed, opponent_type, heuristic_level, model_player, add_noise, temp_threshold = (
        payload
    )
    rng = np.random.default_rng(seed=episode_seed)
    return _play_episode(
        mcts=_WORKER_MCTS,  # type: ignore[arg-type]
        add_noise=add_noise,
        temp_threshold=temp_threshold,
        rng=rng,
        opponent_type=opponent_type,
        opponent_heuristic_level=heuristic_level,
        model_player=model_player,
    )


def _update_stats(stats: dict[str, float | int], winner: int, turn_idx: int) -> None:
    stats["total_turns"] = int(stats["total_turns"]) + turn_idx
    if winner == 1:
        stats["wins_p1"] = int(stats["wins_p1"]) + 1
        return
    if winner == -1:
        stats["wins_p2"] = int(stats["wins_p2"]) + 1
        return
    stats["draws"] = int(stats["draws"]) + 1


def _history_to_examples(
    game_history: list[tuple[np.ndarray, np.ndarray, int]],
    winner: int,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for observation, policy, player_at_turn in game_history:
        if winner == 0:
            z = 0.0
        elif winner == player_at_turn:
            z = 1.0
        else:
            z = -1.0
        examples.append((observation, policy, z))
    return examples


def _play_eval_episode(
    mcts: MCTS,
    rng: np.random.Generator,
    heuristic_level: str,
) -> int:
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard

    board = AtaxxBoard()
    root = None
    model_player = 1 if float(rng.random()) >= 0.5 else -1
    while not board.is_game_over():
        if board.current_player == model_player:
            probs, root = _compute_action_probs(
                board=board,
                mcts=mcts,
                root=root,
                add_noise=False,
                temperature=0.0,
            )
            action_idx = int(np.argmax(probs))
            board.step(ACTION_SPACE.decode(action_idx))
            root = mcts.advance_root(root, action_idx)
            continue
        move = _heuristic_move(board, rng, heuristic_level)
        board.step(move)
        root = mcts.advance_root(root, ACTION_SPACE.encode(move))
    winner = board.get_result()
    if winner == model_player:
        return 1
    if winner == 0:
        return 0
    return -1


def evaluate_model(
    system: AtaxxZero,
    device: str,
    games: int,
    sims: int,
    c_puct: float,
    heuristic_level: str,
    seed: int,
) -> dict[str, float | int]:
    from engine.mcts import MCTS

    system.eval()
    system.to(device)
    mcts = MCTS(
        model=system.model,
        c_puct=c_puct,
        n_simulations=sims,
        device=device,
        use_amp=_cfg_bool("mcts_use_amp"),
    )
    rng = np.random.default_rng(seed=seed)
    wins = 0
    losses = 0
    draws = 0
    for _ in range(games):
        outcome = _play_eval_episode(mcts, rng, heuristic_level)
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1
    score = (wins + 0.5 * draws) / max(1, games)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "heuristic_level": heuristic_level,
        "sims": sims,
    }


def execute_self_play(
    system: AtaxxZero,
    buffer: ReplayBuffer,
    iteration: int,
    device: str,
) -> dict[str, float | int]:
    from engine.mcts import MCTS

    system.eval()
    system.to(device)
    mcts = MCTS(
        model=system.model,
        c_puct=_cfg_float("c_puct"),
        n_simulations=_cfg_int("mcts_sims"),
        device=device,
        use_amp=_cfg_bool("mcts_use_amp"),
    )

    episodes = _cfg_int("episodes_per_iter")
    temp_threshold = _cfg_int("temp_threshold")
    add_noise = _cfg_bool("add_noise")
    rng = np.random.default_rng(seed=_cfg_int("seed") + iteration)
    selfplay_workers = _cfg_int("selfplay_workers")
    _log(f"[Iteration {iteration}] Self-play episodes: {episodes}", verbose_only=True)
    mix_probs, mix_labels = _resolve_opponent_mix()
    mix_text = ", ".join(
        f"{name}={prob:.2f}" for name, prob in zip(mix_labels, mix_probs, strict=True)
    )
    _log(f"  Opponent mix: {mix_text}", verbose_only=True)
    level_probs, level_labels = _resolve_heuristic_level_mix()
    level_text = ", ".join(
        f"{name}={prob:.2f}" for name, prob in zip(level_labels, level_probs, strict=True)
    )
    _log(f"  Heuristic levels: {level_text}", verbose_only=True)

    stats: dict[str, float | int] = {
        "wins_p1": 0,
        "wins_p2": 0,
        "draws": 0,
        "total_turns": 0,
        "avg_game_length": 0.0,
        "episodes_vs_self": 0,
        "episodes_vs_heuristic": 0,
        "episodes_vs_random": 0,
        "episodes_vs_heuristic_easy": 0,
        "episodes_vs_heuristic_normal": 0,
        "episodes_vs_heuristic_hard": 0,
    }

    episode_specs: list[tuple[int, str, str, int]] = []
    for episode_idx in range(episodes):
        opponent_type = _sample_opponent_type(rng)
        heuristic_level = _sample_heuristic_level(rng)
        model_player = 1 if float(rng.random()) >= _cfg_float("model_side_swap_prob") else -1
        episode_seed = _cfg_int("seed") + iteration * 10_000 + episode_idx
        episode_specs.append((episode_seed, opponent_type, heuristic_level, model_player))

    used_parallel = False
    episode_results: list[tuple[str, str, list[tuple[np.ndarray, np.ndarray, int]], int, int]] = []

    if selfplay_workers > 1:
        try:
            max_workers = min(selfplay_workers, episodes)
            worker_payloads = [
                (
                    episode_seed,
                    opponent_type,
                    heuristic_level,
                    model_player,
                    add_noise,
                    temp_threshold,
                )
                for episode_seed, opponent_type, heuristic_level, model_player in episode_specs
            ]
            model_state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in system.model.state_dict().items()
            }
            model_cfg: dict[str, int | float] = {
                "d_model": _cfg_int("d_model"),
                "nhead": _cfg_int("nhead"),
                "num_layers": _cfg_int("num_layers"),
                "dim_feedforward": _cfg_int("dim_feedforward"),
                "dropout": _cfg_float("dropout"),
            }
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_selfplay_process_worker,
                initargs=(
                    model_state_dict,
                    model_cfg,
                    _cfg_float("c_puct"),
                    _cfg_int("mcts_sims"),
                ),
            ) as executor:
                for (
                    (_, opponent_type, heuristic_level, _),
                    episode_result,
                ) in zip(
                    episode_specs,
                    executor.map(_run_episode_in_process_worker, worker_payloads),
                    strict=True,
                ):
                    game_history, winner, turn_idx = episode_result
                    episode_results.append(
                        (opponent_type, heuristic_level, game_history, winner, turn_idx)
                    )
            used_parallel = True
            _log(f"  Self-play process workers active: {max_workers}", verbose_only=True)
        except Exception as exc:
            _log(
                f"  Process self-play failed, falling back to sequential mode: {exc}",
            )
            episode_results.clear()

    if not used_parallel:
        for episode_seed, opponent_type, heuristic_level, model_player in episode_specs:
            local_rng = np.random.default_rng(seed=episode_seed)
            game_history, winner, turn_idx = _play_episode(
                mcts=mcts,
                add_noise=add_noise,
                temp_threshold=temp_threshold,
                rng=local_rng,
                opponent_type=opponent_type,
                opponent_heuristic_level=heuristic_level,
                model_player=model_player,
            )
            episode_results.append((opponent_type, heuristic_level, game_history, winner, turn_idx))

    for episode_idx, (opponent_type, heuristic_level, game_history, winner, turn_idx) in enumerate(
        episode_results,
        start=1,
    ):
        stats[f"episodes_vs_{opponent_type}"] = int(stats[f"episodes_vs_{opponent_type}"]) + 1
        if opponent_type == "heuristic":
            stats[f"episodes_vs_heuristic_{heuristic_level}"] = int(
                stats[f"episodes_vs_heuristic_{heuristic_level}"]
            ) + 1
        _update_stats(stats=stats, winner=winner, turn_idx=turn_idx)
        buffer.save_game(_history_to_examples(game_history=game_history, winner=winner))

        log_every = _cfg_int("episode_log_every")
        if log_every > 0 and episode_idx % log_every == 0:
            _log(
                f"  Episode {episode_idx}/{episodes} | winner={winner} turns={turn_idx}",
                verbose_only=True,
            )

    stats["avg_game_length"] = float(stats["total_turns"]) / float(episodes)
    cache_stats = mcts.cache_stats()
    stats["cache_hits"] = int(cache_stats["hits"])
    stats["cache_misses"] = int(cache_stats["misses"])
    stats["cache_hit_rate"] = float(cache_stats["hit_rate"])
    _log(
        "  Self-play summary: "
        f"P1={stats['wins_p1']} P2={stats['wins_p2']} draws={stats['draws']} "
        f"avg_turns={stats['avg_game_length']:.1f} "
        f"cache_hit={float(stats['cache_hit_rate']):.1%}",
        verbose_only=_is_quiet(),
    )
    return stats


def export_onnx(model: torch.nn.Module, path: str, device: str) -> None:
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, 7, 7, device=device)
    try:
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=path,
            export_params=True,
            opset_version=11,
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes={
                "board": {0: "batch_size"},
                "policy": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )
        _log(f"Exported ONNX to {path}")
    except ModuleNotFoundError as exc:
        _log(
            "ONNX export skipped: missing dependency "
            f"({exc}). Install with: `uv add onnx onnxscript`.",
        )


def main() -> None:
    args = _parse_args()
    _apply_cli_overrides(args)
    _validate_config()
    _ensure_src_on_path()
    from data.dataset import AtaxxDataset, ValidationDataset
    from data.replay_buffer import ReplayBuffer
    from model.system import AtaxxZero

    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Device: {device}")
    trainer_accelerator, trainer_devices, trainer_strategy = _resolve_trainer_hw()
    trainer_precision = _resolve_trainer_precision(trainer_accelerator)
    _log(
        "Trainer HW: "
        f"accelerator={trainer_accelerator}, devices={trainer_devices}, strategy={trainer_strategy}",
    )
    _log(f"Trainer precision: {trainer_precision}")

    checkpoint_dir = Path(_cfg_str("checkpoint_dir"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(_cfg_str("log_dir"))
    log_dir.mkdir(parents=True, exist_ok=True)

    iterations = _cfg_int("iterations")
    epochs = _cfg_int("epochs")
    system = AtaxxZero(
        learning_rate=_cfg_float("learning_rate"),
        weight_decay=_cfg_float("weight_decay"),
        d_model=_cfg_int("d_model"),
        nhead=_cfg_int("nhead"),
        num_layers=_cfg_int("num_layers"),
        dim_feedforward=_cfg_int("dim_feedforward"),
        dropout=_cfg_float("dropout"),
        scheduler_type="cosine",
        max_epochs=iterations * epochs,
    )
    buffer = ReplayBuffer(capacity=_cfg_int("buffer_size"))
    hf_checkpointer = _init_hf_checkpointer()
    hf_upload_executor: ThreadPoolExecutor | None = None
    hf_upload_futures: list[Any] = []
    if hf_checkpointer is not None:
        hf_upload_executor = ThreadPoolExecutor(max_workers=1)
        try:
            start_iteration = hf_checkpointer.load_latest_checkpoint(
                system=system,
                buffer=buffer,
            )
            _log(f"Resumed from HF checkpoint iteration {start_iteration}.")
        except (ValueError, OSError):
            start_iteration = 0
            _log("HF resume failed; starting from scratch.")
    else:
        start_iteration = 0

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="ataxx-epoch{epoch:02d}-val{val_loss:.3f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=str(log_dir), name="ataxx_zero")
    best_eval_score = -1.0

    try:
        for iteration in range(start_iteration + 1, iterations + 1):
            _log(f"=== Iteration {iteration}/{iterations} ===")
            selfplay_start = time.perf_counter()
            selfplay_stats = execute_self_play(
                system=system, buffer=buffer, iteration=iteration, device=device
            )
            selfplay_s = time.perf_counter() - selfplay_start

            train_dataset = AtaxxDataset(
                buffer=buffer, augment=True, reference_buffer=False
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=_cfg_int("batch_size"),
                shuffle=True,
                num_workers=_cfg_int("num_workers"),
                persistent_workers=(
                    _cfg_bool("persistent_workers") and _cfg_int("num_workers") > 0
                ),
                pin_memory=(device == "cuda"),
            )

            val_dataset = ValidationDataset(buffer=buffer, split=_cfg_float("val_split"))
            val_loader = (
                DataLoader(
                    val_dataset,
                    batch_size=_cfg_int("batch_size"),
                    shuffle=False,
                    num_workers=_cfg_int("num_workers"),
                    persistent_workers=(
                        _cfg_bool("persistent_workers") and _cfg_int("num_workers") > 0
                    ),
                    pin_memory=(device == "cuda"),
                )
                if len(val_dataset) > 0
                else None
            )

            trainer = _build_trainer(
                epochs=epochs,
                accelerator=trainer_accelerator,
                devices=trainer_devices,
                strategy=trainer_strategy,
                precision=trainer_precision,
                benchmark=_cfg_bool("trainer_benchmark"),
                checkpoint_callback=checkpoint_callback,
                lr_monitor=lr_monitor,
                logger=logger,
            )
            system.train()
            fit_start = time.perf_counter()
            try:
                trainer.fit(
                    model=system,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )
            except Exception as exc:
                if trainer_devices > 1 and _is_ddp_rendezvous_timeout(exc):
                    _log(
                        "DDP rendezvous failed. Falling back to single-GPU for this run.",
                    )
                    trainer_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
                    trainer_devices = 1
                    trainer_strategy = "auto"
                    trainer_precision = _resolve_trainer_precision(trainer_accelerator)
                    trainer = _build_trainer(
                        epochs=epochs,
                        accelerator=trainer_accelerator,
                        devices=trainer_devices,
                        strategy=trainer_strategy,
                        precision=trainer_precision,
                        benchmark=_cfg_bool("trainer_benchmark"),
                        checkpoint_callback=checkpoint_callback,
                        lr_monitor=lr_monitor,
                        logger=logger,
                    )
                    system.train()
                    trainer.fit(
                        model=system,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader,
                    )
                else:
                    raise
            fit_s = time.perf_counter() - fit_start
            _log(
                f"[Iter {iteration}/{iterations}] selfplay={selfplay_s:.1f}s "
                f"fit={fit_s:.1f}s replay={len(buffer)} "
                f"cache_hit={float(selfplay_stats['cache_hit_rate']):.1%}"
            )

            eval_stats: dict[str, float | int] | None = None
            if _cfg_bool("eval_enabled") and iteration % _cfg_int("eval_every") == 0:
                try:
                    eval_stats = evaluate_model(
                        system=system,
                        device=device,
                        games=_cfg_int("eval_games"),
                        sims=_cfg_int("eval_sims"),
                        c_puct=_cfg_float("c_puct"),
                        heuristic_level=_cfg_str("eval_heuristic_level"),
                        seed=_cfg_int("seed") + 10_000 + iteration,
                    )
                    _log(
                        "Eval summary: "
                        f"W={eval_stats['wins']} L={eval_stats['losses']} D={eval_stats['draws']} "
                        f"score={float(eval_stats['score']):.3f} "
                        f"vs {eval_stats['heuristic_level']} (sims={eval_stats['sims']})"
                    )
                    if float(eval_stats["score"]) > best_eval_score:
                        best_eval_score = float(eval_stats["score"])
                        best_path = checkpoint_dir / "best_eval.ckpt"
                        trainer.save_checkpoint(str(best_path))
                        _log(f"New best checkpoint saved: {best_path}")
                except Exception as exc:
                    _log(f"Eval failed this iteration, continuing training: {exc}")

            save_every = _cfg_int("save_every")
            if iteration % save_every == 0:
                manual_ckpt = checkpoint_dir / f"manual_iter_{iteration:03d}.ckpt"
                try:
                    trainer.save_checkpoint(str(manual_ckpt))
                    _log(f"Saved local checkpoint: {manual_ckpt}")
                    _cleanup_local_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        keep_last_n=_cfg_int("keep_last_n_local_checkpoints"),
                    )
                except OSError:
                    _log("Local checkpoint save failed for this iteration.")

                if hf_checkpointer is not None:
                    try:
                        model_path, buffer_path, metadata_path = hf_checkpointer.save_checkpoint_local(
                            iteration=iteration,
                            system=system,
                            buffer=buffer,
                            config=CONFIG,
                            stats={
                                "replay_size": len(buffer),
                                "best_eval_score": best_eval_score,
                                **(eval_stats or {}),
                            },
                        )
                        if hf_upload_executor is not None:
                            future = hf_upload_executor.submit(
                                hf_checkpointer.upload_checkpoint_files,
                                iteration=iteration,
                                model_path=model_path,
                                buffer_path=buffer_path,
                                metadata_path=metadata_path,
                                keep_last_n=_cfg_int("keep_last_n_hf_checkpoints"),
                            )
                            hf_upload_futures.append(future)
                            _log(f"Queued HF upload for iteration {iteration}.")
                        else:
                            hf_checkpointer.upload_checkpoint_files(
                                iteration=iteration,
                                model_path=model_path,
                                buffer_path=buffer_path,
                                metadata_path=metadata_path,
                                keep_last_n=_cfg_int("keep_last_n_hf_checkpoints"),
                            )
                            _log(f"Uploaded HF checkpoint for iteration {iteration}.")
                    except (OSError, ValueError):
                        _log("HF upload failed for this iteration.")
                if _cfg_bool("export_onnx"):
                    try:
                        export_onnx(system.model, _cfg_str("onnx_path"), device=device)
                    except (OSError, RuntimeError, ValueError):
                        _log("ONNX export failed for this iteration.")
                else:
                    _log("ONNX export disabled by config/CLI.")
                _cleanup_old_log_versions(
                    log_dir=log_dir,
                    run_name="ataxx_zero",
                    keep_last_n=_cfg_int("keep_last_n_log_versions"),
                )
    finally:
        if hf_upload_executor is not None:
            for future in hf_upload_futures:
                try:
                    future.result()
                except Exception:
                    _log("A queued HF upload failed.")
            hf_upload_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
