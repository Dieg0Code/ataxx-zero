from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np

TrainerPrecision = Literal[
    "16",
    "16-mixed",
    "16-true",
    "32",
    "32-true",
    "64",
    "64-true",
    "bf16",
    "bf16-mixed",
    "bf16-true",
    "transformer-engine",
    "transformer-engine-float16",
]


CONFIG: dict[str, int | float | bool | str] = {
    "iterations": 20,
    "episodes_per_iter": 60,
    "mcts_sims": 600,
    "c_puct": 1.5,
    "temp_threshold": 28,
    "add_noise": True,
    "seed": 42,
    "verbose_logs": False,
    "episode_log_every": 25,
    "epochs": 5,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "value_loss_coeff": 0.5,
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
    "hf_run_id": "policy_spatial_v1",
    "hf_token_env": "HF_TOKEN",
    "hf_local_dir": "hf_checkpoints",
    "show_progress_bar": False,
    "trainer_log_every_n_steps": 99_999,
    "monitor_log_every": 5,
    "epoch_pulse_every": 2,
    "num_workers": 4,
    "persistent_workers": True,
    "strict_probs": False,
    "trainer_devices": 1,
    "trainer_strategy": "auto",
    "trainer_precision": "bf16-mixed",
    "trainer_benchmark": True,
    "mcts_use_amp": True,
    "mcts_cache_size": 100_000,
    "mcts_leaf_batch_size": 32,
    "opponent_self_prob": 0.45,
    "opponent_heuristic_prob": 0.5,
    "opponent_random_prob": 0.05,
    "opponent_heuristic_level": "normal",
    "opponent_heuristic_easy_prob": 0.2,
    "opponent_heuristic_normal_prob": 0.5,
    "opponent_heuristic_hard_prob": 0.3,
    "model_side_swap_prob": 0.5,
    "eval_enabled": True,
    "eval_every": 3,
    "eval_games": 12,
    "eval_sims": 400,
    "eval_heuristic_level": "hard",
    "selfplay_workers": 8,
    "compile_model": True,
    "quiet_mode": False,
    "warmup_games": 600,
    "warmup_epochs": 8,
}


def ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ataxx Zero.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--value-loss-coeff", type=float, default=None)
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
    parser.add_argument("--monitor-log-every", type=int, default=None)
    parser.add_argument("--epoch-pulse-every", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--strict-probs", action="store_true")
    parser.add_argument("--no-mcts-amp", action="store_true")
    parser.add_argument("--mcts-cache-size", type=int, default=None)
    parser.add_argument("--mcts-leaf-batch-size", type=int, default=None)
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
    parser.add_argument("--warmup-games", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument(
        "--eval-heuristic-level",
        choices=["easy", "normal", "hard"],
        default=None,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-run-id", default=None)
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
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
    if args.value_loss_coeff is not None:
        CONFIG["value_loss_coeff"] = max(0.0, args.value_loss_coeff)
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
    if args.monitor_log_every is not None:
        CONFIG["monitor_log_every"] = max(1, args.monitor_log_every)
    if args.epoch_pulse_every is not None:
        CONFIG["epoch_pulse_every"] = max(1, args.epoch_pulse_every)
    if args.persistent_workers:
        CONFIG["persistent_workers"] = True
    if args.no_persistent_workers:
        CONFIG["persistent_workers"] = False
    if args.strict_probs:
        CONFIG["strict_probs"] = True
    if args.no_mcts_amp:
        CONFIG["mcts_use_amp"] = False
    if args.mcts_cache_size is not None:
        CONFIG["mcts_cache_size"] = max(0, args.mcts_cache_size)
    if args.mcts_leaf_batch_size is not None:
        CONFIG["mcts_leaf_batch_size"] = max(1, args.mcts_leaf_batch_size)
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
    if args.warmup_games is not None:
        CONFIG["warmup_games"] = max(0, args.warmup_games)
    if args.warmup_epochs is not None:
        CONFIG["warmup_epochs"] = max(0, args.warmup_epochs)
    if args.eval_heuristic_level is not None:
        CONFIG["eval_heuristic_level"] = args.eval_heuristic_level
    if args.quiet:
        CONFIG["show_progress_bar"] = False
        CONFIG["trainer_log_every_n_steps"] = 99_999
        CONFIG["episode_log_every"] = 0
        CONFIG["quiet_mode"] = True
    if args.verbose:
        CONFIG["verbose_logs"] = True
    if args.no_compile:
        CONFIG["compile_model"] = False
    if args.hf:
        CONFIG["hf_enabled"] = True
    if args.hf_repo_id is not None:
        CONFIG["hf_repo_id"] = args.hf_repo_id
    if args.hf_run_id is not None:
        CONFIG["hf_run_id"] = args.hf_run_id.strip()


def cfg_int(key: str) -> int:
    return int(CONFIG[key])


def cfg_float(key: str) -> float:
    return float(CONFIG[key])


def cfg_bool(key: str) -> bool:
    return bool(CONFIG[key])


def cfg_str(key: str) -> str:
    return str(CONFIG[key])


def is_quiet() -> bool:
    return cfg_bool("quiet_mode")


def log(message: str, verbose_only: bool = False) -> None:
    if verbose_only and not cfg_bool("verbose_logs"):
        return
    print(message)


def validate_config() -> None:
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
        "mcts_leaf_batch_size",
        "monitor_log_every",
        "epoch_pulse_every",
    )
    for key in int_positive_keys:
        if cfg_int(key) <= 0:
            raise ValueError(f"CONFIG['{key}'] must be > 0, got {cfg_int(key)}.")
    if cfg_int("num_workers") < 0:
        raise ValueError("CONFIG['num_workers'] must be >= 0.")
    if cfg_int("warmup_games") < 0:
        raise ValueError("CONFIG['warmup_games'] must be >= 0.")
    if cfg_int("warmup_epochs") < 0:
        raise ValueError("CONFIG['warmup_epochs'] must be >= 0.")
    if cfg_float("value_loss_coeff") < 0.0:
        raise ValueError("CONFIG['value_loss_coeff'] must be >= 0.")
    if cfg_int("mcts_cache_size") < 0:
        raise ValueError("CONFIG['mcts_cache_size'] must be >= 0.")

    opp_sum = (
        cfg_float("opponent_self_prob")
        + cfg_float("opponent_heuristic_prob")
        + cfg_float("opponent_random_prob")
    )
    heu_sum = (
        cfg_float("opponent_heuristic_easy_prob")
        + cfg_float("opponent_heuristic_normal_prob")
        + cfg_float("opponent_heuristic_hard_prob")
    )
    if cfg_bool("strict_probs"):
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
    if not cfg_bool("strict_probs") and not np.isclose(opp_sum, 1.0, atol=1e-6):
        log(
            f"Opponent probs sum to {opp_sum:.6f}; they will be normalized automatically.",
            verbose_only=True,
        )
    if not cfg_bool("strict_probs") and not np.isclose(heu_sum, 1.0, atol=1e-6):
        log(
            f"Heuristic level probs sum to {heu_sum:.6f}; they will be normalized automatically.",
            verbose_only=True,
        )


def resolve_trainer_precision(accelerator: str) -> TrainerPrecision:
    configured = cast(TrainerPrecision, cfg_str("trainer_precision"))
    if accelerator == "gpu":
        return configured
    return cast(TrainerPrecision, "32-true")
