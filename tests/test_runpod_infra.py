from __future__ import annotations

import unittest

from training.runpod_infra import (
    build_pod_env,
    build_train_start_command,
    pod_finished,
)


class TestRunPodInfra(unittest.TestCase):
    def test_build_pod_env_keeps_required_keys(self) -> None:
        sample_value = "example"
        env = build_pod_env(
            hf_token=sample_value,
            hf_repo_id="dieg0code/ataxx-zero",
            hf_run_id="policy_spatial_v1",
        )
        self.assertEqual(env[0]["key"], "PYTHONUNBUFFERED")
        self.assertEqual(env[1]["key"], "HF_TOKEN")
        self.assertEqual(env[1]["value"], sample_value)
        self.assertEqual(env[2]["value"], "dieg0code/ataxx-zero")
        self.assertEqual(env[3]["value"], "policy_spatial_v1")

    def test_build_train_start_command_contains_pinned_ref_and_hf_namespace(self) -> None:
        cmd = build_train_start_command(
            repository="dieg0code/ataxx-zero",
            git_ref="abc123def",
            train_args="--iterations 5 --save-every 2",
            hf_repo_id="dieg0code/ataxx-zero",
            hf_run_id="policy_spatial_v1",
        )
        self.assertIn("git checkout abc123def", cmd)
        self.assertIn("--hf-repo-id dieg0code/ataxx-zero", cmd)
        self.assertIn("--hf-run-id policy_spatial_v1", cmd)

    def test_pod_finished_for_terminal_desired_status(self) -> None:
        self.assertTrue(pod_finished(desired_status="STOPPED", runtime_status="RUNNING"))

    def test_pod_finished_for_terminal_runtime_status(self) -> None:
        self.assertTrue(pod_finished(desired_status="RUNNING", runtime_status="FAILED"))

    def test_pod_finished_for_non_terminal_status(self) -> None:
        self.assertFalse(pod_finished(desired_status="RUNNING", runtime_status="RUNNING"))


if __name__ == "__main__":
    unittest.main()
