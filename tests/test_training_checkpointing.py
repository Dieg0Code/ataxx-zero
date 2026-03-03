from __future__ import annotations

import unittest
from concurrent.futures import Future

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
