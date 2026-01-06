import pytest

from src.application.dto.prediction_request import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.entities.model import Model
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


@pytest.mark.asyncio
async def test_predict_handler_success():
    # Setup
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    handler = PredictHandler(repo, engine)
    
    # Register a model
    model = Model(
        id="sentiment-model",
        version="v1",
        uri="local://models/sentiment-v1.onnx",
        framework="onnx"
    )
    await repo.save(model)
    
    # Execute command
    command = PredictCommand(
        model_id="sentiment-model",
        model_version="v1",
        features=[0.1, 0.2, 0.3]
    )
    
    prediction = await handler.execute(command)
    
    # Verify
    assert prediction.model_id == "sentiment-model"
    assert prediction.result == pytest.approx(0.2)  # noqa: PLR2004
    assert prediction.confidence.value == 0.99  # noqa: PLR2004
