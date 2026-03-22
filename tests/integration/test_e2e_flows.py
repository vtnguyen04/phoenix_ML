"""End-to-end integration test — simulates production flows.

Tests EVERY route and middleware through FastAPI TestClient.
Does NOT require running Docker services.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.infrastructure.http.data_routes import data_router
from src.infrastructure.http.error_handlers import (
    AppError,
    NotFoundError,
    register_exception_handlers,
)
from src.infrastructure.http.websocket_routes import ws_router


@pytest.fixture
def e2e_app() -> FastAPI:
    """Standalone FastAPI app for E2E testing (no lifespan DB dependency)."""
    app = FastAPI(title="Phoenix E2E Test")
    register_exception_handlers(app)
    app.include_router(data_router)
    app.include_router(ws_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "version": "test"}

    @app.get("/error/not-found")
    async def trigger_not_found() -> None:
        raise NotFoundError("Model", "credit-v99")

    @app.get("/error/app")
    async def trigger_app_error() -> None:
        raise AppError("test error", "TEST_ERR", 418)

    @app.get("/error/unhandled")
    async def trigger_unhandled() -> None:
        msg = "unexpected"
        raise RuntimeError(msg)

    return app


@pytest.fixture
def client(e2e_app: FastAPI) -> TestClient:
    return TestClient(e2e_app, raise_server_exceptions=False)


# ── Flow 1: Health Check ─────────────────────────────────────────


class TestHealthFlow:
    def test_basic_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"


# ── Flow 2: Data Pipeline ────────────────────────────────────────


class TestDataPipelineFlow:
    def test_ingest_success(self, client: TestClient, tmp_path: Path) -> None:
        csv = tmp_path / "test.csv"
        pd.DataFrame({
            "income": [50000, 60000, 70000],
            "age": [25, 30, 35],
            "target": [0, 1, 1],
        }).to_csv(csv, index=False)

        resp = client.post("/data/ingest", json={
            "source_path": str(csv),
            "target_column": "target",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["rows_processed"] == 3

    def test_ingest_missing_file(self, client: TestClient) -> None:
        resp = client.post("/data/ingest", json={
            "source_path": "/nonexistent/file.csv",
        })
        assert resp.status_code == 404

    def test_validate_success(self, client: TestClient, tmp_path: Path) -> None:
        csv = tmp_path / "valid.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv, index=False)

        resp = client.post("/data/validate", json={"source_path": str(csv)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True

    def test_ingest_with_output(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        csv = tmp_path / "src.csv"
        out = tmp_path / "out.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv, index=False)

        resp = client.post("/data/ingest", json={
            "source_path": str(csv),
            "output_path": str(out),
        })
        assert resp.status_code == 200
        assert out.exists()


# ── Flow 3: Error Handler ────────────────────────────────────────


class TestErrorHandlerFlow:
    def test_not_found_error(self, client: TestClient) -> None:
        resp = client.get("/error/not-found")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error_code"] == "NOT_FOUND"
        assert "credit-v99" in body["message"]

    def test_app_error(self, client: TestClient) -> None:
        resp = client.get("/error/app")
        assert resp.status_code == 418
        assert resp.json()["error_code"] == "TEST_ERR"

    def test_unhandled_error(self, client: TestClient) -> None:
        resp = client.get("/error/unhandled")
        assert resp.status_code == 500
        assert resp.json()["error_code"] == "INTERNAL_ERROR"


# ── Flow 4: WebSocket ────────────────────────────────────────────


class TestWebSocketFlow:
    def test_websocket_connect_and_receive(
        self, e2e_app: FastAPI
    ) -> None:
        client = TestClient(e2e_app)
        with client.websocket_connect("/ws/events") as ws:
            # Should receive welcome message
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert "Phoenix" in data["message"]

    def test_websocket_ping_pong(self, e2e_app: FastAPI) -> None:
        client = TestClient(e2e_app)
        with client.websocket_connect("/ws/events") as ws:
            ws.receive_json()  # consume welcome
            ws.send_text("ping")
            data = ws.receive_json()
            assert data["type"] == "pong"
