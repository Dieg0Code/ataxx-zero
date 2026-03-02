from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.deps.inference import get_inference_service_dep


class TestApiInferenceDep(unittest.TestCase):
    @patch("api.deps.inference._build_inference_service", side_effect=ModuleNotFoundError("hf_xet"))
    @patch("api.deps.inference.resolve_artifact_uri", side_effect=lambda value: value)
    def test_maps_module_not_found_to_http_503(self, *_: object) -> None:
        with self.assertRaises(HTTPException) as ctx:
            get_inference_service_dep()

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Inference service unavailable", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()

