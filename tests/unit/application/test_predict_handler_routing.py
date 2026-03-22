from unittest.mock import AsyncMock

import pytest

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.application.handlers.predict_handler import PredictHandler
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.services.inference_service import InferenceService
from phoenix_ml.domain.inference.value_objects.confidence_score import ConfidenceScore
from phoenix_ml.domain.shared.event_bus import DomainEventBus


@pytest.fixture
def mock_inference_service() -> AsyncMock:
    return AsyncMock(spec=InferenceService)


@pytest.mark.asyncio
async def test_predict_handler_routing_logic(mock_inference_service: AsyncMock) -> None:
    handler = PredictHandler(mock_inference_service, DomainEventBus())

    # Mock Inference Service result
    mock_inference_service.predict.return_value = Prediction(
        model_id="m1",
        model_version="v2",
        result=1,
        confidence=ConfidenceScore(value=0.9),
        latency_ms=1.0,
    )

    # Command WITHOUT version
    cmd = PredictCommand(model_id="m1", features=[1.0, 2.0])

    # Execute
    prediction = await handler.execute(cmd)

    # Verify
    assert prediction.model_version == "v2"
    mock_inference_service.predict.assert_called_once()


@pytest.mark.asyncio
async def test_predict_handler_routing_no_candidates(
    mock_inference_service: AsyncMock,
) -> None:
    handler = PredictHandler(mock_inference_service, DomainEventBus())

    # Mock Inference Service raising error
    mock_inference_service.predict.side_effect = ValueError("No active versions found")

    cmd = PredictCommand(model_id="m1", features=[1.0])

    with pytest.raises(ValueError, match="No active versions found"):
        await handler.execute(cmd)
