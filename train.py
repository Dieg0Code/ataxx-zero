from __future__ import annotations

import argparse
import json
import os
import sys
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
    "keep_last_n_hf_checkpoints": 3,
    "onnx_path": "ataxx_model.onnx",
    "export_onnx": True,
    "hf_enabled": False,
    "hf_repo_id": "",
    "hf_token_env": "HF_TOKEN",
    "hf_local_dir": "hf_checkpoints",
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
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


def _log(message: str, verbose_only: bool = False) -> None:
    if verbose_only and not _cfg_bool("verbose_logs"):
        return
    print(message)


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

    def save_checkpoint(
        self,
        *,
        iteration: int,
        system: AtaxxZero,
        buffer: ReplayBuffer,
        config: dict[str, int | float | bool | str],
        stats: dict[str, float | int],
    ) -> None:
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
        if len(all_examples) > 0:
            observations = np.asarray([ex[0] for ex in all_examples], dtype=np.float32)
            policies = np.asarray([ex[1] for ex in all_examples], dtype=np.float32)
            values = np.asarray([ex[2] for ex in all_examples], dtype=np.float32)
            np.savez_compressed(
                self.local_dir / buffer_name,
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
        (self.local_dir / metadata_name).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(self.local_dir),
            commit_message=f"Checkpoint iteration {iteration}",
        )

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
    add_noise: bool,
    temperature: float,
) -> np.ndarray:
    from game.actions import ACTION_SPACE

    probs = mcts.run(
        board=board,
        add_dirichlet_noise=add_noise,
        temperature=temperature,
    )
    total_prob = float(np.sum(probs))
    if total_prob > 0.0:
        return probs

    valid_moves = board.get_valid_moves()
    fallback = ACTION_SPACE.mask_from_moves(
        valid_moves,
        include_pass=(len(valid_moves) == 0),
    )
    return fallback / float(np.sum(fallback))


def _select_action_idx(
    probs: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature > 0.0:
        return int(rng.choice(len(probs), p=probs))
    return int(np.argmax(probs))


def _play_episode(
    mcts: MCTS,
    add_noise: bool,
    temp_threshold: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], int, int]:
    from game.actions import ACTION_SPACE
    from game.board import AtaxxBoard

    board = AtaxxBoard()
    game_history: list[tuple[np.ndarray, np.ndarray, int]] = []
    turn_idx = 0

    while not board.is_game_over():
        turn_idx += 1
        temperature = 1.0 if turn_idx <= temp_threshold else 0.0
        probs = _compute_action_probs(
            board=board,
            mcts=mcts,
            add_noise=add_noise,
            temperature=temperature,
        )
        game_history.append(
            (
                board.get_observation(),
                probs.astype(np.float32),
                board.current_player,
            )
        )
        action_idx = _select_action_idx(probs=probs, temperature=temperature, rng=rng)
        board.step(ACTION_SPACE.decode(action_idx))

    return game_history, board.get_result(), turn_idx


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
    )

    episodes = _cfg_int("episodes_per_iter")
    temp_threshold = _cfg_int("temp_threshold")
    add_noise = _cfg_bool("add_noise")
    rng = np.random.default_rng(seed=_cfg_int("seed"))
    _log(f"[Iteration {iteration}] Self-play episodes: {episodes}")

    stats: dict[str, float | int] = {
        "wins_p1": 0,
        "wins_p2": 0,
        "draws": 0,
        "total_turns": 0,
        "avg_game_length": 0.0,
    }

    for episode_idx in range(episodes):
        game_history, winner, turn_idx = _play_episode(
            mcts=mcts,
            add_noise=add_noise,
            temp_threshold=temp_threshold,
            rng=rng,
        )
        _update_stats(stats=stats, winner=winner, turn_idx=turn_idx)
        buffer.save_game(_history_to_examples(game_history=game_history, winner=winner))

        log_every = _cfg_int("episode_log_every")
        if log_every > 0 and (episode_idx + 1) % log_every == 0:
            _log(
                f"  Episode {episode_idx + 1}/{episodes} | winner={winner} turns={turn_idx}",
                verbose_only=True,
            )

    stats["avg_game_length"] = float(stats["total_turns"]) / float(episodes)
    _log(
        "  Self-play summary: "
        f"P1={stats['wins_p1']} P2={stats['wins_p2']} draws={stats['draws']} "
        f"avg_turns={stats['avg_game_length']:.1f}"
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
    _ensure_src_on_path()
    from data.dataset import AtaxxDataset, ValidationDataset
    from data.replay_buffer import ReplayBuffer
    from model.system import AtaxxZero

    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Device: {device}")

    checkpoint_dir = Path(_cfg_str("checkpoint_dir"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    if hf_checkpointer is not None:
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
    logger = TensorBoardLogger(save_dir=_cfg_str("log_dir"), name="ataxx_zero")

    for iteration in range(start_iteration + 1, iterations + 1):
        _log(f"=== Iteration {iteration}/{iterations} ===")
        execute_self_play(
            system=system, buffer=buffer, iteration=iteration, device=device
        )
        _log(f"Replay size: {len(buffer)}")

        train_dataset = AtaxxDataset(
            buffer=buffer, augment=True, reference_buffer=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=_cfg_int("batch_size"),
            shuffle=True,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        val_dataset = ValidationDataset(buffer=buffer, split=_cfg_float("val_split"))
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=_cfg_int("batch_size"),
                shuffle=False,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
            if len(val_dataset) > 0
            else None
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
        )
        trainer.fit(
            model=system,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        save_every = _cfg_int("save_every")
        if iteration % save_every == 0:
            manual_ckpt = checkpoint_dir / f"manual_iter_{iteration:03d}.ckpt"
            try:
                trainer.save_checkpoint(str(manual_ckpt))
                _log(f"Saved local checkpoint: {manual_ckpt}")
            except OSError:
                _log("Local checkpoint save failed for this iteration.")

            if hf_checkpointer is not None:
                try:
                    hf_checkpointer.save_checkpoint(
                        iteration=iteration,
                        system=system,
                        buffer=buffer,
                        config=CONFIG,
                        stats={"replay_size": len(buffer)},
                    )
                    hf_checkpointer.cleanup_local_checkpoints(
                        keep_last_n=_cfg_int("keep_last_n_hf_checkpoints")
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


if __name__ == "__main__":
    main()
