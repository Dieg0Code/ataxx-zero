from __future__ import annotations

import unittest

from training.config_runtime import CONFIG
from training.selfplay_runtime import handle_parallel_selfplay_failure


class TestTrainingSelfplayRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_handle_parallel_selfplay_failure_raises_in_fail_fast_mode(self) -> None:
        CONFIG["fail_on_selfplay_parallel_error"] = True
        with self.assertRaises(RuntimeError):
            handle_parallel_selfplay_failure(RuntimeError("pool broke"))

    def test_handle_parallel_selfplay_failure_allows_fallback_when_configured(self) -> None:
        CONFIG["fail_on_selfplay_parallel_error"] = False
        handle_parallel_selfplay_failure(RuntimeError("pool broke"))


if __name__ == "__main__":
    unittest.main()
