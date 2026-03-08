from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from pytorch_lightning.loggers import CSVLogger

from training.logging_runtime import build_training_logger


class TestTrainingLoggingRuntime(unittest.TestCase):
    def test_build_training_logger_falls_back_to_csv_when_tensorboard_missing(self) -> None:
        with patch(
            "training.logging_runtime.TensorBoardLogger",
            side_effect=ModuleNotFoundError("tensorboard missing"),
        ):
            logger = build_training_logger(Path(".tmp_test_train_logger"))
        self.assertIsInstance(logger, CSVLogger)


if __name__ == "__main__":
    unittest.main()
