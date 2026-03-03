from __future__ import annotations

import unittest
from datetime import timedelta

from pytorch_lightning.strategies import DDPStrategy

from training.config_runtime import CONFIG
from training.trainer_runtime import resolve_trainer_strategy


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


if __name__ == "__main__":
    unittest.main()
