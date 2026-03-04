from __future__ import annotations

import sys
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

from training.checkpointing import (
    HuggingFaceCheckpointer,
    drain_completed_hf_uploads,
    ensure_hf_ready,
    should_save_iteration_checkpoint,
    wait_for_hf_uploads,
)
from training.config_runtime import CONFIG


class TestTrainingCheckpointing(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_repo_path_is_namespaced_by_run_id(self) -> None:
        checkpointer = object.__new__(HuggingFaceCheckpointer)
        checkpointer.run_id = "policy_spatial_v1"
        repo_path = checkpointer._repo_path("model_iter_040.pt")
        self.assertEqual(repo_path, "runs/policy_spatial_v1/model_iter_040.pt")

    def test_repo_path_for_run_allows_explicit_source_namespace(self) -> None:
        repo_path = HuggingFaceCheckpointer._repo_path_for_run(
            run_id="policy_spatial_v2",
            filename="model_iter_001.pt",
        )
        self.assertEqual(repo_path, "runs/policy_spatial_v2/model_iter_001.pt")

    def test_load_latest_checkpoint_can_bootstrap_from_explicit_run_without_buffer(self) -> None:
        sample_value = "sample_value"
        checkpointer = object.__new__(HuggingFaceCheckpointer)
        checkpointer.repo_id = "dieg0code/ataxx-zero"
        checkpointer.token = sample_value
        checkpointer.run_id = "policy_target_v2"
        checkpointer.local_dir = Path()
        checkpointer.api = Mock()
        checkpointer.api.list_repo_files.return_value = [
            "runs/policy_source_v1/model_iter_022.pt",
            "runs/policy_source_v1/buffer_iter_022.npz",
            "runs/policy_target_v2/model_iter_010.pt",
        ]

        hf_download_mock = MagicMock(return_value="model_iter_022.pt")
        hub_module = cast(Any, types.ModuleType("huggingface_hub"))
        hub_module.hf_hub_download = hf_download_mock

        system = Mock()
        buffer = Mock()

        with patch.dict(sys.modules, {"huggingface_hub": hub_module}), patch(
            "training.checkpointing.torch.load"
        ) as torch_load_mock:
            torch_load_mock.return_value = {"state_dict": {}}
            loaded_iter = checkpointer.load_latest_checkpoint(
                system=system,
                buffer=buffer,
                run_id="policy_source_v1",
                load_buffer=False,
            )

        self.assertEqual(loaded_iter, 22)
        hf_download_mock.assert_called_once_with(
            repo_id="dieg0code/ataxx-zero",
            filename="runs/policy_source_v1/model_iter_022.pt",
            repo_type="model",
            token=sample_value,
            local_dir=".",
        )
        system.load_state_dict.assert_called_once_with({})
        buffer.clear.assert_not_called()
        buffer.save_game.assert_not_called()

    def test_ensure_hf_ready_raises_when_hf_enabled_without_checkpointer(self) -> None:
        CONFIG["hf_enabled"] = True
        CONFIG["hf_token_env"] = "HF_TOKEN"  # noqa: S105 - test fixture value, not a secret.
        with self.assertRaises(RuntimeError):
            ensure_hf_ready(None)

    def test_ensure_hf_ready_noop_when_hf_disabled(self) -> None:
        CONFIG["hf_enabled"] = False
        ensure_hf_ready(None)

    def test_drain_completed_hf_uploads_keeps_pending_only(self) -> None:
        done: Future[None] = Future()
        done.set_result(None)
        pending: Future[None] = Future()
        remaining = drain_completed_hf_uploads([done, pending], fail_on_error=True)
        self.assertEqual(remaining, [pending])

    def test_drain_completed_hf_uploads_raises_when_fail_fast_enabled(self) -> None:
        failed: Future[None] = Future()
        failed.set_exception(RuntimeError("upload failed"))
        with self.assertRaises(RuntimeError):
            drain_completed_hf_uploads([failed], fail_on_error=True)

    def test_drain_completed_hf_uploads_continues_when_fail_fast_disabled(self) -> None:
        failed: Future[None] = Future()
        failed.set_exception(RuntimeError("upload failed"))
        remaining = drain_completed_hf_uploads([failed], fail_on_error=False)
        self.assertEqual(remaining, [])

    def test_wait_for_hf_uploads_raises_timeout_in_fail_fast_mode(self) -> None:
        pending: Future[None] = Future()
        with self.assertRaises(RuntimeError):
            wait_for_hf_uploads([pending], timeout_s=1.0, fail_on_error=True)

    def test_wait_for_hf_uploads_tolerates_timeout_when_configured(self) -> None:
        pending: Future[None] = Future()
        wait_for_hf_uploads([pending], timeout_s=1.0, fail_on_error=False)

    def test_should_save_iteration_checkpoint_on_schedule(self) -> None:
        self.assertTrue(
            should_save_iteration_checkpoint(
                iteration=6,
                total_iterations=40,
                save_every=3,
            )
        )

    def test_should_save_iteration_checkpoint_on_last_iteration(self) -> None:
        self.assertTrue(
            should_save_iteration_checkpoint(
                iteration=40,
                total_iterations=40,
                save_every=3,
            )
        )

    def test_should_not_save_iteration_checkpoint_off_schedule_non_final(self) -> None:
        self.assertFalse(
            should_save_iteration_checkpoint(
                iteration=7,
                total_iterations=40,
                save_every=3,
            )
        )


if __name__ == "__main__":
    unittest.main()
