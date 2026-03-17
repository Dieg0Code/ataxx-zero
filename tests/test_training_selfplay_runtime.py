from __future__ import annotations

import unittest
from unittest.mock import patch

from training.config_runtime import CONFIG
from training.selfplay_runtime import (
    handle_parallel_selfplay_failure,
    resolve_parallel_selfplay_worker_devices,
)


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

    def test_resolve_parallel_selfplay_worker_devices_keeps_cpu_pool(self) -> None:
        worker_devices = resolve_parallel_selfplay_worker_devices(
            device="cpu",
            requested_workers=3,
            episodes=5,
        )
        self.assertEqual(worker_devices, ["cpu", "cpu", "cpu"])

    @patch("training.selfplay_runtime.torch.cuda.device_count", return_value=1)
    @patch("training.selfplay_runtime.torch.cuda.is_available", return_value=True)
    def test_resolve_parallel_selfplay_worker_devices_disables_single_gpu_pool(
        self,
        _mock_is_available: object,
        _mock_device_count: object,
    ) -> None:
        worker_devices = resolve_parallel_selfplay_worker_devices(
            device="cuda",
            requested_workers=4,
            episodes=8,
        )
        self.assertEqual(worker_devices, [])

    @patch("training.selfplay_runtime.torch.cuda.device_count", return_value=2)
    @patch("training.selfplay_runtime.torch.cuda.is_available", return_value=True)
    def test_resolve_parallel_selfplay_worker_devices_uses_available_gpus(
        self,
        _mock_is_available: object,
        _mock_device_count: object,
    ) -> None:
        worker_devices = resolve_parallel_selfplay_worker_devices(
            device="cuda",
            requested_workers=4,
            episodes=8,
        )
        self.assertEqual(worker_devices, ["cuda:0", "cuda:1"])


if __name__ == "__main__":
    unittest.main()
