"""
Integration Test: gRPC InferenceServicer.

Tests the gRPC service layer directly (without network transport),
verifying Predict and HealthCheck RPCs perform correctly.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.value_objects.confidence_score import ConfidenceScore
from src.infrastructure.http.grpc_server import (
    HealthCheckResponse,
    InferenceServicer,
    PredictRequest,
)


@pytest.fixture
def mock_predict_handler() -> AsyncMock:
    handler = AsyncMock()
    handler.execute = AsyncMock(
        return_value=Prediction(
            model_id="credit-risk",
            model_version="v1",
            result=1,
            confidence=ConfidenceScore(value=0.92),
            latency_ms=12.5,
        )
    )
    return handler


@pytest.fixture
def servicer(mock_predict_handler: AsyncMock) -> InferenceServicer:
    return InferenceServicer(predict_handler=mock_predict_handler)


@pytest.fixture
def mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


class TestGRPCInferenceServicer:
    """Integration tests for the gRPC InferenceServicer."""

    async def test_predict_returns_valid_response(
        self,
        servicer: InferenceServicer,
        mock_context: MagicMock,
    ) -> None:
        """Predict RPC returns a valid response with prediction data."""
        request = PredictRequest(
            model_id="credit-risk",
            model_version="v1",
            entity_id="customer-001",
            features=[0.5] * 30,
        )

        response = await servicer.Predict(request, mock_context)

        assert response.model_id == "credit-risk"
        assert response.version == "v1"
        assert response.result == [1.0]
        assert response.confidence == pytest.approx(0.92)
        assert response.latency_ms == pytest.approx(12.5)
        assert response.prediction_id != ""

    async def test_predict_with_minimal_request(
        self,
        servicer: InferenceServicer,
        mock_context: MagicMock,
    ) -> None:
        """Predict succeeds with only model_id."""
        request = PredictRequest(model_id="credit-risk")
        response = await servicer.Predict(request, mock_context)

        assert response.model_id == "credit-risk"
        mock_context.set_code.assert_not_called()

    async def test_predict_handles_value_error(
        self,
        mock_predict_handler: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        """ValueError from handler maps to NOT_FOUND gRPC status."""
        import grpc  # noqa: PLC0415

        mock_predict_handler.execute = AsyncMock(
            side_effect=ValueError("Model not found")
        )
        servicer = InferenceServicer(predict_handler=mock_predict_handler)

        request = PredictRequest(model_id="nonexistent")
        await servicer.Predict(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        mock_context.set_details.assert_called_once_with("Model not found")

    async def test_predict_handles_unexpected_error(
        self,
        mock_predict_handler: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        """Unexpected exception maps to INTERNAL gRPC status."""
        import grpc  # noqa: PLC0415

        mock_predict_handler.execute = AsyncMock(
            side_effect=RuntimeError("ONNX engine crash")
        )
        servicer = InferenceServicer(predict_handler=mock_predict_handler)

        request = PredictRequest(model_id="credit-risk")
        await servicer.Predict(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

    async def test_health_check_returns_serving(
        self,
        servicer: InferenceServicer,
        mock_context: MagicMock,
    ) -> None:
        """HealthCheck RPC returns SERVING status."""
        response = await servicer.HealthCheck({}, mock_context)

        assert response.status == HealthCheckResponse.SERVING
