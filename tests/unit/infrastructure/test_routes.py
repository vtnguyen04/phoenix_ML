"""Tests for main API routes using httpx AsyncClient with DI overrides."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from phoenix_ml.application.handlers.predict_handler import PredictHandler
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.value_objects.confidence_score import ConfidenceScore
from phoenix_ml.infrastructure.http.dependencies import get_predict_handler
from phoenix_ml.infrastructure.http.routes import router
from phoenix_ml.infrastructure.persistence.database import get_db


def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    # Override DB dependency
    mock_session = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.add = MagicMock()

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    # Override predict handler
    mock_handler = MagicMock(spec=PredictHandler)
    mock_handler.execute = AsyncMock(
        return_value=Prediction(
            model_id="m1",
            model_version="v1",
            result=1,
            confidence=ConfidenceScore(value=0.95),
            latency_ms=2.5,
        )
    )

    async def override_handler():  # type: ignore[no-untyped-def]
        return mock_handler

    app.dependency_overrides[get_predict_handler] = override_handler

    return app


@pytest.fixture
def app() -> FastAPI:
    return _create_test_app()


async def test_health_check(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


async def test_predict_success(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"model_id": "m1", "features": [1.0, 2.0, 3.0]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "m1"
        assert data["result"] == 1
        assert data["confidence"]["value"] == 0.95


async def test_feedback(app: FastAPI) -> None:
    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/feedback",
            json={"prediction_id": "pred-1", "ground_truth": 1},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "feedback_received"


async def test_get_model_not_found(app: FastAPI) -> None:
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/models/nonexistent")
        assert resp.status_code == 404


async def test_register_model(app: FastAPI) -> None:
    mock_session = MagicMock()
    mock_session.merge = AsyncMock()
    mock_session.commit = AsyncMock()

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/models/register",
            json={
                "model_id": "new-model",
                "version": "v1",
                "uri": "s3://bucket/model.onnx",
                "framework": "onnx",
                "stage": "challenger",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "new-model"
        assert data["status"] == "registered"


async def test_rollback_challengers(app: FastAPI) -> None:
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.commit = AsyncMock()

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/models/rollback", json={"model_id": "m1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "m1"


async def test_get_drift_reports(app: FastAPI) -> None:
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/monitoring/reports/m1")
        assert resp.status_code == 200
        assert resp.json() == []


async def test_get_model_performance(app: FastAPI) -> None:
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def override_db():  # type: ignore[no-untyped-def]
        yield mock_session

    app.dependency_overrides[get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/monitoring/performance/m1")
        assert resp.status_code == 200
