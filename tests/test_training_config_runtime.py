from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from training.config_runtime import CONFIG, apply_cli_overrides, parse_args


class TestTrainingConfigRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_hf_bootstrap_flags_are_applied_from_cli(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--hf",
                "--hf-run-id",
                "policy_target_v2",
                "--hf-bootstrap-run-id",
                "policy_source_v1",
                "--hf-reset-iteration",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertTrue(bool(CONFIG["hf_enabled"]))
        self.assertEqual(str(CONFIG["hf_run_id"]), "policy_target_v2")
        self.assertEqual(str(CONFIG["hf_bootstrap_run_id"]), "policy_source_v1")
        self.assertTrue(bool(CONFIG["hf_reset_iteration"]))


if __name__ == "__main__":
    unittest.main()
