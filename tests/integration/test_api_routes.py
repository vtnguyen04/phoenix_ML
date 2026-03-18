"""
API Route Integration Tests — Tests the FastAPI endpoints with mocked dependencies.
Fills the critical test gap: verifies route wiring, request/response schemas, and error handling.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.value_objects.confidence_score import ConfidenceScore
from src.infrastructure.http.fastapi_server import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_and_version(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client: TestClient) -> None:
        from src.infrastructure.http.dependencies import get_predict_handler  # noqa: PLC0415
        from src.infrastructure.persistence.database import get_db  # noqa: PLC0415

        mock_prediction = Prediction(
            model_id="credit-risk",
            model_version="v1",
            result=1,
            confidence=ConfidenceScore(value=0.85),
            latency_ms=0.42,
        )
        mock_handler = AsyncMock()
        mock_handler.execute = AsyncMock(return_value=mock_prediction)

        async def override_handler() -> AsyncMock:
            return mock_handler

        async def override_db() -> AsyncMock:
            return AsyncMock()

        app.dependency_overrides[get_predict_handler] = override_handler
        app.dependency_overrides[get_db] = override_db

        try:
            response = client.post(
                "/predict",
                json={
                    "model_id": "credit-risk",
                    "entity_id": "customer-0001",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == "credit-risk"
            assert data["version"] == "v1"
            assert data["result"] == 1
            assert data["confidence"]["value"] == 0.85
            assert "prediction_id" in data
            assert "latency_ms" in data
        finally:
            app.dependency_overrides.clear()

    def test_predict_missing_model_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/predict", content="not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestFeedbackEndpoint:
    """Tests for POST /feedback."""

    def test_feedback_schema_validation(self, client: TestClient) -> None:
        # Missing prediction_id should fail
        response = client.post("/feedback", json={"ground_truth": 1})
        assert response.status_code == 422


class TestDriftEndpoint:
    """Tests for GET /monitoring/drift/{model_id}."""

    def test_drift_requires_model_id(self, client: TestClient) -> None:
        response = client.get("/monitoring/drift/")
        # FastAPI returns 404 for missing path param (not a valid route)
        assert response.status_code in (404, 405)


class TestPerformanceEndpoint:
    """Tests for GET /monitoring/performance/{model_id}."""

    def test_performance_requires_model_id(self, client: TestClient) -> None:
        response = client.get("/monitoring/performance/")
        assert response.status_code in (404, 405)


class TestReportsEndpoint:
    """Tests for GET /monitoring/reports/{model_id}."""

    def test_reports_requires_model_id(self, client: TestClient) -> None:
        response = client.get("/monitoring/reports/")
        assert response.status_code in (404, 405)
