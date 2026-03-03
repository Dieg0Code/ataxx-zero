from __future__ import annotations

import unittest
from datetime import timedelta
from unittest.mock import patch

from pytorch_lightning.strategies import DDPStrategy

from training.config_runtime import CONFIG
from training.trainer_runtime import resolve_trainer_hw, resolve_trainer_strategy


class TestTrainingTrainerRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_resolve_trainer_strategy_ddp_uses_configured_timeout(self) -> None:
        CONFIG["ddp_timeout_seconds"] = 75
        resolved = resolve_trainer_strategy("ddp")
        self.assertIsInstance(resolved, DDPStrategy)
        strategy = resolved
        self.assertEqual(strategy._timeout, timedelta(seconds=75))
        self.assertEqual(strategy._start_method, "popen")

    def test_resolve_trainer_strategy_ddp_spawn_uses_spawn_start_method(self) -> None:
        CONFIG["ddp_timeout_seconds"] = 90
        resolved = resolve_trainer_strategy("ddp_spawn")
        self.assertIsInstance(resolved, DDPStrategy)
        strategy = resolved
        self.assertEqual(strategy._timeout, timedelta(seconds=90))
        self.assertEqual(strategy._start_method, "spawn")

    def test_resolve_trainer_strategy_passthrough_for_auto(self) -> None:
        resolved = resolve_trainer_strategy("auto")
        self.assertEqual(resolved, "auto")

    def test_resolve_trainer_hw_downgrades_ddp_spawn_when_one_gpu_available(self) -> None:
        CONFIG["trainer_devices"] = 2
        CONFIG["trainer_strategy"] = "ddp_spawn"
        with (
            patch("training.trainer_runtime.torch.cuda.is_available", return_value=True),
            patch("training.trainer_runtime.torch.cuda.device_count", return_value=1),
        ):
            accelerator, devices, strategy = resolve_trainer_hw()
        self.assertEqual(accelerator, "gpu")
        self.assertEqual(devices, 1)
        self.assertEqual(strategy, "auto")

    def test_resolve_trainer_hw_keeps_ddp_spawn_when_two_gpus_available(self) -> None:
        CONFIG["trainer_devices"] = 2
        CONFIG["trainer_strategy"] = "ddp_spawn"
        with (
            patch("training.trainer_runtime.torch.cuda.is_available", return_value=True),
            patch("training.trainer_runtime.torch.cuda.device_count", return_value=2),
        ):
            accelerator, devices, strategy = resolve_trainer_hw()
        self.assertEqual(accelerator, "gpu")
        self.assertEqual(devices, 2)
        self.assertEqual(strategy, "ddp_spawn")


if __name__ == "__main__":
    unittest.main()
