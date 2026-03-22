"""WebSocket real-time event streaming.

Broadcasts: predictions, drift alerts, model updates, system events.
Clients connect to /ws/events to receive live JSON messages.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
ws_router = APIRouter()


class EventType(str, Enum):
    PREDICTION = "prediction"
    DRIFT_ALERT = "drift_alert"
    MODEL_UPDATE = "model_update"
    HEALTH_CHANGE = "health_change"
    SYSTEM = "system"


class ConnectionManager:
    """Manage WebSocket connections for broadcasting events."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(
            "WebSocket connected: %s (%d total)",
            websocket.client,
            len(self._connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(
            "WebSocket disconnected (%d remaining)",
            len(self._connections),
        )

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Send event to all connected clients."""
        message = json.dumps(event, default=str)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


# Global connection manager (singleton)
manager = ConnectionManager()


async def emit_event(
    event_type: EventType,
    data: dict[str, Any],
) -> None:
    """Emit a real-time event to all WebSocket clients."""
    event = {
        "type": event_type.value,
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data,
    }
    await manager.broadcast(event)


@ws_router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time event streaming."""
    await manager.connect(websocket)
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Phoenix ML real-time events",
            "timestamp": datetime.now(UTC).isoformat(),
        })
        # Keep connection alive, listen for client messages
        while True:
            data = await websocket.receive_text()
            # Echo ping/pong for keep-alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
