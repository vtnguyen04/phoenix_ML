from unittest.mock import AsyncMock, Mock

import pytest

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.services.routing_strategy import RoutingStrategy
from src.domain.inference.value_objects.confidence_score import ConfidenceScore
from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage


@pytest.fixture
def mock_components():
    repo = AsyncMock(spec=ModelRepository)
    engine = AsyncMock(spec=InferenceEngine)
    fs = AsyncMock(spec=FeatureStore)
    storage = AsyncMock(spec=ArtifactStorage)
    routing = Mock(spec=RoutingStrategy)
    return repo, engine, fs, storage, routing

@pytest.mark.asyncio
async def test_predict_handler_routing_logic(mock_components) -> None:
    repo, engine, fs, storage, routing = mock_components
    handler = PredictHandler(repo, engine, fs, storage, routing)

    # Setup
    model_v1 = Model(id="m1", version="v1", uri="loc://v1", framework="onnx")
    model_v2 = Model(id="m1", version="v2", uri="loc://v2", framework="onnx")
    
    # Mock Repository returning active versions
    repo.get_active_versions.return_value = [model_v1, model_v2]
    
    # Mock Routing selecting v2
    routing.select_model.return_value = model_v2
    
    # Mock Engine Prediction
    engine.predict.return_value = Prediction(
        model_id="m1", 
        model_version="v2", 
        result=1, 
        confidence=ConfidenceScore(value=0.9), 
        latency_ms=1.0
    )

    # Command WITHOUT version
    cmd = PredictCommand(model_id="m1", features=[1.0, 2.0])
    
    # Execute
    prediction = await handler.execute(cmd)

    # Verify
    assert prediction.model_version == "v2"
    repo.get_active_versions.assert_awaited_once_with("m1")
    routing.select_model.assert_called_once()
    engine.load.assert_awaited_with(model_v2)

@pytest.mark.asyncio
async def test_predict_handler_routing_no_candidates(mock_components) -> None:
    repo, engine, fs, storage, routing = mock_components
    handler = PredictHandler(repo, engine, fs, storage, routing)

    # Mock Repository returning NO versions
    repo.get_active_versions.return_value = []

    cmd = PredictCommand(model_id="m1", features=[1.0])
    
    with pytest.raises(ValueError, match="No active versions found"):
        await handler.execute(cmd)
