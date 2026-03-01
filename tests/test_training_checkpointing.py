from __future__ import annotations

import unittest

from training.checkpointing import (
    HuggingFaceCheckpointer,
    should_save_iteration_checkpoint,
)


class TestTrainingCheckpointing(unittest.TestCase):
    def test_repo_path_is_namespaced_by_run_id(self) -> None:
        checkpointer = object.__new__(HuggingFaceCheckpointer)
        checkpointer.run_id = "policy_spatial_v1"
        repo_path = checkpointer._repo_path("model_iter_040.pt")
        self.assertEqual(repo_path, "runs/policy_spatial_v1/model_iter_040.pt")

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
