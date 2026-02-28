from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any
from uuid import UUID

from fastapi import WebSocket


class GameplayWsHub:
    def __init__(self) -> None:
        self._connections: dict[UUID, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, game_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[game_id].add(websocket)

    async def disconnect(self, game_id: UUID, websocket: WebSocket) -> None:
        async with self._lock:
            peers = self._connections.get(game_id)
            if peers is None:
                return
            peers.discard(websocket)
            if len(peers) == 0:
                self._connections.pop(game_id, None)

    async def send_personal(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        await websocket.send_json(payload)

    async def broadcast(self, game_id: UUID, payload: dict[str, Any]) -> None:
        async with self._lock:
            peers = list(self._connections.get(game_id, ()))
        if not peers:
            return
        stale: list[WebSocket] = []
        for websocket in peers:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        if stale:
            async with self._lock:
                current = self._connections.get(game_id)
                if current is None:
                    return
                for websocket in stale:
                    current.discard(websocket)
                if len(current) == 0:
                    self._connections.pop(game_id, None)


gameplay_ws_hub = GameplayWsHub()
