from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.config import Settings


class TestApiOpenApiAuth(unittest.TestCase):
    def test_openapi_includes_bearer_auth_scheme(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (OpenAPI Test)",
            app_env="test",
            app_docs_enabled=True,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        response = client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()

        components = schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        self.assertIn("bearerAuth", security_schemes)
        scheme = security_schemes["bearerAuth"]
        self.assertEqual(scheme.get("type"), "http")
        self.assertEqual(scheme.get("scheme"), "bearer")


if __name__ == "__main__":
    unittest.main()
