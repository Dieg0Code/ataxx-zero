from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.deps.inference import get_inference_service_dep, preload_inference_service


class TestApiInferenceDep(unittest.TestCase):
    @patch("api.deps.inference._build_inference_service", side_effect=ModuleNotFoundError("hf_xet"))
    @patch("api.deps.inference.resolve_artifact_uri", side_effect=lambda value: value)
    def test_maps_module_not_found_to_http_503(self, *_: object) -> None:
        with self.assertRaises(HTTPException) as ctx:
            get_inference_service_dep()

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Inference service unavailable", str(ctx.exception.detail))

    @patch("api.deps.inference.get_inference_service_dep")
    def test_preload_inference_service_warms_up_once(self, get_dep: Mock) -> None:
        service = Mock()
        get_dep.return_value = service

        resolved = preload_inference_service()

        self.assertIs(resolved, service)
        service.warmup.assert_called_once_with(mode="fast")

    @patch(
        "api.deps.inference.get_inference_service_dep",
        side_effect=HTTPException(status_code=503, detail="inference unavailable"),
    )
    def test_preload_inference_service_returns_none_when_unavailable(self, *_: object) -> None:
        self.assertIsNone(preload_inference_service())


if __name__ == "__main__":
    unittest.main()
