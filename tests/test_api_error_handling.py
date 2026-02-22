from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.error_handling import register_error_handlers


class EchoPayload(BaseModel):
    name: str


def build_test_app() -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise HTTPException(status_code=404, detail="Resource missing")

    @app.post("/echo")
    def echo(payload: EchoPayload) -> dict[str, str]:
        return {"name": payload.name}

    @app.get("/db-down")
    def db_down() -> None:
        raise OperationalError("SELECT 1", {}, OSError("connection refused"))

    return app


class TestApiErrorHandling(unittest.TestCase):
    def test_http_exception_uses_standard_shape(self) -> None:
        client = TestClient(build_test_app())
        response = client.get("/boom", headers={"X-Request-ID": "req-123"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers.get("X-Request-ID"), "req-123")

        payload = response.json()
        self.assertEqual(payload["error_code"], "not_found")
        self.assertEqual(payload["message"], "Resource missing")
        self.assertEqual(payload["detail"], "Resource missing")
        self.assertEqual(payload["request_id"], "req-123")

    def test_validation_error_uses_standard_shape(self) -> None:
        client = TestClient(build_test_app())
        response = client.post("/echo", json={})

        self.assertEqual(response.status_code, 422)
        self.assertIn("X-Request-ID", response.headers)

        payload = response.json()
        self.assertEqual(payload["error_code"], "validation_error")
        self.assertEqual(payload["message"], "Validation failed")
        self.assertEqual(payload["detail"], "Validation failed")
        self.assertIn("details", payload)
        self.assertTrue(isinstance(payload["details"], list))
        self.assertTrue(payload["request_id"])

    def test_db_unavailable_maps_to_503(self) -> None:
        client = TestClient(build_test_app())
        response = client.get("/db-down")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error_code"], "service_unavailable")
        self.assertEqual(payload["message"], "Database unavailable")
        self.assertEqual(payload["detail"], "Database unavailable")
        self.assertIn("request_id", payload)


if __name__ == "__main__":
    unittest.main()
