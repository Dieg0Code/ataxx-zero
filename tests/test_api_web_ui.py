from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.config import Settings


class TestApiWebUi(unittest.TestCase):
    def test_root_redirects_to_web(self) -> None:
        app = create_app(
            settings=Settings(app_env="test", app_docs_enabled=False, app_log_requests=False)
        )
        client = TestClient(app)

        response = client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/web")

    def test_web_page_and_static_assets_are_served(self) -> None:
        app = create_app(
            settings=Settings(app_env="test", app_docs_enabled=False, app_log_requests=False)
        )
        client = TestClient(app)

        html_response = client.get("/web")
        self.assertEqual(html_response.status_code, 200)
        self.assertIn("ATAXX ARENA", html_response.text)
        self.assertIn("/web/static/app.js", html_response.text)

        js_response = client.get("/web/static/app.js")
        self.assertEqual(js_response.status_code, 200)
        self.assertIn("function aiTurn()", js_response.text)

        css_response = client.get("/web/static/styles.css")
        self.assertEqual(css_response.status_code, 200)
        self.assertIn("--p1", css_response.text)


if __name__ == "__main__":
    unittest.main()
