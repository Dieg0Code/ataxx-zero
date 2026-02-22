from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.config import Settings


class TestApiHealth(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (Test)",
            app_env="test",
            app_docs_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["app"], "Ataxx Zero API (Test)")
        self.assertEqual(payload["env"], "test")

    def test_docs_disabled_hides_docs_routes(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (Test)",
            app_env="test",
            app_docs_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        docs_response = client.get("/docs")
        redoc_response = client.get("/redoc")
        self.assertEqual(docs_response.status_code, 404)
        self.assertEqual(redoc_response.status_code, 404)

    def test_ready_endpoint_returns_ready_when_db_check_passes(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (Test)",
            app_env="test",
            app_docs_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)
        with patch(
            "api.modules.health.router._check_db_ready",
            new=AsyncMock(return_value=True),
        ):
            response = client.get("/health/ready")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["checks"]["db"], True)

    def test_ready_endpoint_returns_503_when_db_check_fails(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (Test)",
            app_env="test",
            app_docs_enabled=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)
        with patch(
            "api.modules.health.router._check_db_ready",
            new=AsyncMock(return_value=False),
        ):
            response = client.get("/health/ready")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error_code"], "service_unavailable")
        self.assertEqual(payload["detail"], "Database not ready")

    def test_cors_headers_are_present_when_origins_configured(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (Test)",
            app_env="test",
            app_docs_enabled=False,
            app_cors_origins=["http://localhost:5173"],
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_request_logging_emits_api_request_log(self) -> None:
        settings = Settings(
            app_name="Ataxx Zero API (Test)",
            app_env="test",
            app_docs_enabled=False,
            app_log_requests=True,
            app_log_json=False,
        )
        app = create_app(settings=settings)
        client = TestClient(app)

        with self.assertLogs("api.request", level="INFO") as captured:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        joined = "\n".join(captured.output)
        self.assertIn("request_completed", joined)


if __name__ == "__main__":
    unittest.main()
