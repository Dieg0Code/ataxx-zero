from __future__ import annotations

import re
import runpy
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CI_TRAIN_PATHS = [
    "train.py",
    "src/training",
    "src/engine",
    "src/model",
    "src/game",
    "src/data",
    "tests/test_mcts_numerics.py",
    "tests/test_training_bootstrap.py",
    "tests/test_training_checkpointing.py",
    "tests/test_training_curriculum.py",
    "tests/test_training_monitor.py",
    "tests/test_training_step_numerics.py",
    "scripts/export_model_onnx.py",
    "scripts/check_onnx_parity.py",
]
SKIP_GLOBS = [".venv/**", "**/__pycache__/**", ".git/**"]


def _load_line_policy() -> tuple[int, Callable[..., list[tuple[str, int]]]]:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"max_python_file_lines\s*=\s*(\d+)", pyproject_text)
    if match is None:
        raise AssertionError("Missing tool.repo_policy.max_python_file_lines in pyproject.toml.")
    namespace = runpy.run_path(
        str((ROOT / "scripts" / "check_python_max_lines.py").resolve()),
        run_name="python_line_policy_test",
    )
    collect = cast(
        Callable[..., list[tuple[str, int]]],
        namespace["_collect_violations"],
    )
    return int(match.group(1)), collect


class TestTrainingPythonFileLengthPolicy(unittest.TestCase):
    def test_ci_train_scope_respects_python_file_limit(self) -> None:
        max_lines, collect_violations = _load_line_policy()
        violations = collect_violations(
            roots=[ROOT / path for path in CI_TRAIN_PATHS],
            max_lines=max_lines,
            skip_globs=SKIP_GLOBS,
        )
        self.assertEqual([], violations, msg=f"Python file length policy violations: {violations}")


if __name__ == "__main__":
    unittest.main()
