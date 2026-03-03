from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi import HTTPException, status
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.app import create_app
from api.config import Settings
from api.deps.inference import get_inference_service_dep
from api.modules.gameplay.schemas import MoveRequest
from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.serialization import board_to_state
from inference.service import InferenceResult


class _StubInferenceService:
    def __init__(self, move: tuple[int, int, int, int] | None) -> None:
        self._move = move

    def predict(self, board: AtaxxBoard, mode: str = "fast") -> InferenceResult:
        del board
        return InferenceResult(
            move=self._move,
            action_idx=ACTION_SPACE.pass_index
            if self._move is None
            else ACTION_SPACE.encode(self._move),
            value=0.42,
            mode="fast" if mode == "fast" else "strong",
        )


class TestApiMove(unittest.TestCase):
    def test_move_endpoint_returns_predicted_move(self) -> None:
        app = create_app()
        stub = _StubInferenceService((0, 0, 1, 1))
        app.dependency_overrides[get_inference_service_dep] = lambda: stub
        client = TestClient(app)

        board = AtaxxBoard()
        payload = MoveRequest(board=board_to_state(board), mode="fast").model_dump()
        response = client.post("/api/v1/gameplay/move", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["mode"], "fast")
        self.assertEqual(body["move"], {"r1": 0, "c1": 0, "r2": 1, "c2": 1})

    def test_move_endpoint_supports_pass(self) -> None:
        app = create_app()
        stub = _StubInferenceService(None)
        app.dependency_overrides[get_inference_service_dep] = lambda: stub
        client = TestClient(app)

        board = AtaxxBoard()
        payload = MoveRequest(board=board_to_state(board), mode="strong").model_dump()
        response = client.post("/api/v1/gameplay/move", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["mode"], "strong")
        self.assertIsNone(body["move"])
        self.assertEqual(body["action_idx"], ACTION_SPACE.pass_index)

    def test_move_endpoint_rejects_invalid_board(self) -> None:
        app = create_app()
        stub = _StubInferenceService((0, 0, 1, 1))
        app.dependency_overrides[get_inference_service_dep] = lambda: stub
        client = TestClient(app)

        bad_payload = {
            "board": {
                "grid": [[0] * 7 for _ in range(6)],  # invalid shape
                "current_player": 1,
                "half_moves": 0,
            },
            "mode": "fast",
        }
        response = client.post("/api/v1/gameplay/move", json=bad_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid board payload", response.json()["detail"])

    def test_move_endpoint_supports_heuristic_without_inference_service(self) -> None:
        app = create_app()

        def _unavailable_inference() -> _StubInferenceService:
            raise AssertionError("Inference dependency should not be resolved for heuristic modes.")

        app.dependency_overrides[get_inference_service_dep] = _unavailable_inference
        client = TestClient(app)

        board = AtaxxBoard()
        payload = MoveRequest(board=board_to_state(board), mode="heuristic_normal").model_dump()
        response = client.post("/api/v1/gameplay/move", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["mode"], "heuristic_normal")
        self.assertIsInstance(body["action_idx"], int)

    def test_move_endpoint_supports_new_heuristic_levels(self) -> None:
        app = create_app()
        client = TestClient(app)
        board = AtaxxBoard()

        for mode in ("heuristic_apex", "heuristic_gambit", "heuristic_sentinel"):
            payload = MoveRequest(board=board_to_state(board), mode=mode).model_dump()
            response = client.post("/api/v1/gameplay/move", json=payload)
            self.assertEqual(response.status_code, 200, msg=f"unexpected failure for mode={mode}")
            body = response.json()
            self.assertEqual(body["mode"], mode)
            self.assertIsInstance(body["action_idx"], int)

    def test_move_endpoint_falls_back_to_heuristic_when_inference_is_unavailable(self) -> None:
        app = create_app()

        def _unavailable_inference() -> _StubInferenceService:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference unavailable in test",
            )

        app.dependency_overrides[get_inference_service_dep] = _unavailable_inference
        client = TestClient(app)

        board = AtaxxBoard()
        payload = MoveRequest(board=board_to_state(board), mode="fast").model_dump()
        response = client.post("/api/v1/gameplay/move", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["mode"], "heuristic_easy")
        self.assertIsInstance(body["action_idx"], int)

    def test_move_endpoint_fallback_level_honors_settings(self) -> None:
        app = create_app(settings=Settings(inference_fallback_heuristic_level="apex"))

        def _unavailable_inference() -> _StubInferenceService:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference unavailable in test",
            )

        app.dependency_overrides[get_inference_service_dep] = _unavailable_inference
        client = TestClient(app)

        board = AtaxxBoard()
        payload = MoveRequest(board=board_to_state(board), mode="fast").model_dump()
        response = client.post("/api/v1/gameplay/move", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["mode"], "heuristic_apex")
        self.assertIsInstance(body["action_idx"], int)


if __name__ == "__main__":
    unittest.main()
