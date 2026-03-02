from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.inference_artifacts import resolve_artifact_uri


class TestApiInferenceArtifacts(unittest.TestCase):
    def test_resolve_local_path_returns_same_value(self) -> None:
        path = resolve_artifact_uri("checkpoints/model_iter_039.pt")
        self.assertEqual(path, "checkpoints/model_iter_039.pt")

    def test_resolve_file_uri_returns_local_path(self) -> None:
        path = resolve_artifact_uri("file:///var/lib/ataxx/model.ckpt")
        self.assertEqual(path, "/var/lib/ataxx/model.ckpt")

    @patch("api.inference_artifacts.importlib.import_module")
    def test_resolve_hf_uri_downloads_artifact(self, mock_download: MagicMock) -> None:
        hf_download = MagicMock(return_value="/var/lib/ataxx/model_iter_039.pt")
        mock_download.return_value = SimpleNamespace(hf_hub_download=hf_download)

        resolved = resolve_artifact_uri("hf://dieg0code/ataxx-zero/model_iter_039.pt")
        self.assertEqual(resolved, "/var/lib/ataxx/model_iter_039.pt")
        hf_download.assert_called_once_with(
            repo_id="dieg0code/ataxx-zero",
            filename="model_iter_039.pt",
            revision=None,
            token=None,
        )

    def test_rejects_unknown_scheme(self) -> None:
        with self.assertRaises(ValueError):
            resolve_artifact_uri("s3://bucket/model.pt")


if __name__ == "__main__":
    unittest.main()
