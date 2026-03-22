"""
Integration Test: gRPC InferenceServicer.

Tests the gRPC service layer directly (without network transport),
verifying Predict and HealthCheck RPCs perform correctly using
the compiled proto stubs.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from phoenix_ml.config import get_settings
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.value_objects.confidence_score import ConfidenceScore
from phoenix_ml.infrastructure.grpc.grpc_server import InferenceServicer
from phoenix_ml.infrastructure.grpc.proto import inference_pb2

_settings = get_settings()


@pytest.fixture
def mock_predict_handler() -> AsyncMock:
    handler = AsyncMock()
    handler.execute = AsyncMock(
        return_value=Prediction(
            model_id=_settings.DEFAULT_MODEL_ID,
            model_version=_settings.DEFAULT_MODEL_VERSION,
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
    """Integration tests for the gRPC InferenceServicer with proto stubs."""

    async def test_predict_returns_valid_response(
        self,
        servicer: InferenceServicer,
        mock_context: MagicMock,
    ) -> None:
        """Predict RPC returns a valid proto response with prediction data."""
        request = inference_pb2.PredictRequest(  # type: ignore[attr-defined]
            model_id=_settings.DEFAULT_MODEL_ID,
            model_version=_settings.DEFAULT_MODEL_VERSION,
            entity_id="customer-001",
            features=[0.5] * 30,
        )

        response = await servicer.Predict(request, mock_context)

        assert response.model_id == _settings.DEFAULT_MODEL_ID
        assert response.version == _settings.DEFAULT_MODEL_VERSION
        assert list(response.result) == [1.0]
        assert response.confidence == pytest.approx(0.92, abs=0.01)
        assert response.latency_ms == pytest.approx(12.5)
        assert response.prediction_id != ""

    async def test_predict_with_minimal_request(
        self,
        servicer: InferenceServicer,
        mock_context: MagicMock,
    ) -> None:
        """Predict succeeds with only model_id."""
        request = inference_pb2.PredictRequest(model_id=_settings.DEFAULT_MODEL_ID)  # type: ignore[attr-defined]
        response = await servicer.Predict(request, mock_context)

        assert response.model_id == _settings.DEFAULT_MODEL_ID
        mock_context.set_code.assert_not_called()

    async def test_predict_handles_value_error(
        self,
        mock_predict_handler: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        """ValueError from handler maps to NOT_FOUND gRPC status."""
        import grpc  # noqa: PLC0415

        from phoenix_ml.infrastructure.grpc.grpc_server import InferenceServicer  # noqa: PLC0415

        mock_predict_handler.execute = AsyncMock(side_effect=ValueError("Model not found"))
        servicer = InferenceServicer(predict_handler=mock_predict_handler)

        request = inference_pb2.PredictRequest(model_id="nonexistent")  # type: ignore[attr-defined]
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

        from phoenix_ml.infrastructure.grpc.grpc_server import InferenceServicer  # noqa: PLC0415

        mock_predict_handler.execute = AsyncMock(side_effect=RuntimeError("ONNX engine crash"))
        servicer = InferenceServicer(predict_handler=mock_predict_handler)

        request = inference_pb2.PredictRequest(model_id=_settings.DEFAULT_MODEL_ID)  # type: ignore[attr-defined]
        await servicer.Predict(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

    async def test_health_check_returns_serving(
        self,
        servicer: InferenceServicer,
        mock_context: MagicMock,
    ) -> None:
        """HealthCheck RPC returns SERVING status."""
        request = inference_pb2.HealthCheckRequest()  # type: ignore[attr-defined]
        response = await servicer.HealthCheck(request, mock_context)

        assert response.status == inference_pb2.HealthCheckResponse.SERVING  # type: ignore[attr-defined]
