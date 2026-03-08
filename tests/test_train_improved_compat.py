from __future__ import annotations

import runpy
import unittest
from pathlib import Path


class TestTrainImprovedCompat(unittest.TestCase):
    def test_wrapper_reexports_notebook_training_api(self) -> None:
        namespace = runpy.run_path(
            str(Path("train_improved.py").resolve()),
            run_name="train_improved_test",
        )

        self.assertIn("CONFIG", namespace)
        self.assertIn("main", namespace)
        self.assertIn("parse_args", namespace)
        self.assertTrue(callable(namespace["main"]))
        self.assertTrue(callable(namespace["parse_args"]))


if __name__ == "__main__":
    unittest.main()
