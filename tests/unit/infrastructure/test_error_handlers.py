"""Tests for global error handlers."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.infrastructure.http.error_handlers import (
    AppError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    register_exception_handlers,
)


@pytest.fixture
def error_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/app-error")
    async def raise_app_error() -> None:
        raise AppError("custom error", error_code="CUSTOM", status_code=418)

    @app.get("/not-found")
    async def raise_not_found() -> None:
        raise NotFoundError("Model", "credit-risk-v99")

    @app.get("/validation")
    async def raise_validation() -> None:
        raise ValidationError("Invalid input", {"field": "age"})

    @app.get("/auth")
    async def raise_auth() -> None:
        raise AuthenticationError()

    @app.get("/rate-limit")
    async def raise_rate_limit() -> None:
        raise RateLimitError()

    @app.get("/http-error")
    async def raise_http() -> None:
        raise HTTPException(status_code=404, detail="Not found")

    @app.get("/unhandled")
    async def raise_unhandled() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    return app


@pytest.fixture
def client(error_app: FastAPI) -> TestClient:
    return TestClient(error_app, raise_server_exceptions=False)


class TestErrorHandlers:
    def test_app_error(self, client: TestClient) -> None:
        resp = client.get("/app-error")
        assert resp.status_code == 418
        body = resp.json()
        assert body["error_code"] == "CUSTOM"
        assert "custom error" in body["message"]

    def test_not_found(self, client: TestClient) -> None:
        resp = client.get("/not-found")
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "NOT_FOUND"

    def test_validation(self, client: TestClient) -> None:
        resp = client.get("/validation")
        assert resp.status_code == 422
        body = resp.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["details"]["field"] == "age"

    def test_auth(self, client: TestClient) -> None:
        resp = client.get("/auth")
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTHENTICATION_ERROR"

    def test_rate_limit(self, client: TestClient) -> None:
        resp = client.get("/rate-limit")
        assert resp.status_code == 429
        assert resp.json()["error_code"] == "RATE_LIMIT_EXCEEDED"

    def test_http_exception(self, client: TestClient) -> None:
        resp = client.get("/http-error")
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "NOT_FOUND"

    def test_unhandled(self, client: TestClient) -> None:
        resp = client.get("/unhandled")
        assert resp.status_code == 500
        assert resp.json()["error_code"] == "INTERNAL_ERROR"
