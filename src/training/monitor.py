from __future__ import annotations

from collections.abc import Mapping

import torch


def _to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


class TrainingMonitor:
    """Compact, iteration-level logger for long training runs."""

    def __init__(self, total_iterations: int, log_every: int = 5) -> None:
        self.total = max(1, int(total_iterations))
        self.log_every = max(1, int(log_every))
        self.best_eval_score = -1.0

    def _prefix(self, iteration: int) -> str:
        return f"[{iteration:02d}/{self.total}]"

    def log_warmup(self, *, examples: int, games: int) -> None:
        print(f"[00/{self.total}] warmup examples={examples} games={games}")

    def log_epoch(
        self,
        *,
        iteration: int,
        epoch: int,
        max_epochs: int,
        logged_metrics: Mapping[str, object],
    ) -> None:
        loss = _to_float(logged_metrics.get("train/loss"))
        val_loss = _to_float(logged_metrics.get("val/loss"))
        pol_acc = _to_float(logged_metrics.get("train/policy_accuracy"))
        print(
            f"{self._prefix(iteration)} ep={epoch}/{max_epochs} "
            f"loss={loss:.3f} val={val_loss:.3f} pol_acc={pol_acc:.1%}",
        )

    def log_iteration(
        self,
        *,
        iteration: int,
        selfplay_s: float,
        fit_s: float,
        buffer_size: int,
        selfplay_stats: Mapping[str, float | int],
        logged_metrics: Mapping[str, object],
    ) -> None:
        loss = _to_float(logged_metrics.get("train/loss"))
        loss_v = _to_float(logged_metrics.get("train/loss_value"))
        cache_hr = float(selfplay_stats.get("cache_hit_rate", 0.0))
        print(
            f"{self._prefix(iteration)} "
            f"sp={selfplay_s:.0f}s tr={fit_s:.0f}s "
            f"buf={buffer_size} "
            f"loss={loss:.3f}|{loss_v:.3f} "
            f"cache={cache_hr:.0%}",
        )

        self._log_alerts(
            iteration=iteration,
            loss=loss,
            loss_v=loss_v,
            cache_hr=cache_hr,
            avg_turns=float(selfplay_stats.get("avg_game_length", 0.0)),
        )

        if iteration % self.log_every != 0:
            return

        wins = int(selfplay_stats.get("wins_p1", 0))
        losses = int(selfplay_stats.get("wins_p2", 0))
        draws = int(selfplay_stats.get("draws", 0))
        avg_turns = float(selfplay_stats.get("avg_game_length", 0.0))

        ep_self = int(selfplay_stats.get("episodes_vs_self", 0))
        ep_heu = int(selfplay_stats.get("episodes_vs_heuristic", 0))
        ep_rnd = int(selfplay_stats.get("episodes_vs_random", 0))
        total_ep = max(1, ep_self + ep_heu + ep_rnd)

        ep_easy = int(selfplay_stats.get("episodes_vs_heuristic_easy", 0))
        ep_normal = int(selfplay_stats.get("episodes_vs_heuristic_normal", 0))
        ep_hard = int(selfplay_stats.get("episodes_vs_heuristic_hard", 0))
        total_heu = max(1, ep_easy + ep_normal + ep_hard)

        pol_acc = _to_float(logged_metrics.get("train/policy_accuracy"))
        val_loss = _to_float(logged_metrics.get("val/loss"))
        print(
            "         "
            f"W/L/D={wins}/{losses}/{draws} avg_turns={avg_turns:.0f} "
            f"vs=self:{ep_self/total_ep:.0%} heu:{ep_heu/total_ep:.0%} rnd:{ep_rnd/total_ep:.0%}",
        )
        print(
            "         "
            f"heuristic easy:{ep_easy/total_heu:.0%} normal:{ep_normal/total_heu:.0%} hard:{ep_hard/total_heu:.0%} "
            f"pol_acc={pol_acc:.1%} val_loss={val_loss:.3f}",
        )

    def _log_alerts(
        self,
        *,
        iteration: int,
        loss: float,
        loss_v: float,
        cache_hr: float,
        avg_turns: float,
    ) -> None:
        if loss != loss:
            self.log_warning(iteration=iteration, message="loss es NaN, revisar gradientes/LR.")
        if loss > 0.0 and loss_v > (loss * 0.7):
            self.log_warning(
                iteration=iteration,
                message=f"value loss domina ({loss_v:.3f}/{loss:.3f}), revisar value_loss_coeff.",
            )
        if avg_turns > 85.0:
            self.log_warning(
                iteration=iteration,
                message=f"avg_turns={avg_turns:.0f}, posible oscilacion/repeticion.",
            )
        if iteration > 5 and cache_hr < 0.15:
            self.log_warning(
                iteration=iteration,
                message=f"cache hit rate baja ({cache_hr:.0%}), ajustar mcts_cache_size.",
            )

    def log_eval(self, *, iteration: int, eval_stats: Mapping[str, float | int | str]) -> bool:
        score = float(eval_stats.get("score", 0.0))
        wins = int(eval_stats.get("wins", 0))
        losses = int(eval_stats.get("losses", 0))
        draws = int(eval_stats.get("draws", 0))
        level = str(eval_stats.get("heuristic_level", "unknown"))
        sims = int(eval_stats.get("sims", 0))
        is_best = score > self.best_eval_score
        if is_best:
            self.best_eval_score = score
        suffix = " *** BEST ***" if is_best else ""
        print(
            f"{self._prefix(iteration)} EVAL "
            f"W={wins} L={losses} D={draws} "
            f"score={score:.3f} vs {level} (sims={sims}){suffix}",
        )
        return is_best

    def log_checkpoint(self, *, iteration: int, path: str) -> None:
        print(f"{self._prefix(iteration)} CKPT saved -> {path}")

    def log_warning(self, *, iteration: int, message: str) -> None:
        print(f"{self._prefix(iteration)} WARN {message}")
