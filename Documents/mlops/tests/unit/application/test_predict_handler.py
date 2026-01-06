import pytest

from src.application.dto.prediction_request import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.entities.model import Model
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


@pytest.mark.asyncio
async def test_predict_handler_with_direct_features():
    # Setup
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    fs = InMemoryFeatureStore()
    handler = PredictHandler(repo, engine, fs)
    
    # Register model
    model = Model(id="m1", version="v1", uri="loc", framework="onnx")
    await repo.save(model)
    
    # Execute with explicit features
    command = PredictCommand(
        model_id="m1", model_version="v1", features=[1.0, 2.0]
    )
    prediction = await handler.execute(command)
    assert prediction.result == pytest.approx(1.5) # Mean of [1.0, 2.0]

@pytest.mark.asyncio
async def test_predict_handler_with_feature_store():
    # Setup
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    fs = InMemoryFeatureStore()
    handler = PredictHandler(repo, engine, fs)
    
    # Register model & Seed features
    model = Model(id="m2", version="v1", uri="loc", framework="onnx")
    await repo.save(model)
    fs.add_features("user-1", {"f1": 10.0, "f2": 20.0, "f3": 30.0})
    
    # Execute with entity_id
    command = PredictCommand(
        model_id="m2", model_version="v1", entity_id="user-1"
    )
    prediction = await handler.execute(command)
    
    # Expect mean of [10, 20, 30] = 20.0
    assert prediction.result == pytest.approx(20.0)

@pytest.mark.asyncio
async def test_predict_handler_missing_features_raises_error():
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    fs = InMemoryFeatureStore()
    handler = PredictHandler(repo, engine, fs)
    
    model = Model(id="m3", version="v1", uri="loc", framework="onnx")
    await repo.save(model)
    
    # No features, no entity_id
    command = PredictCommand(model_id="m3", model_version="v1")
    
    with pytest.raises(ValueError, match="No features provided"):
        await handler.execute(command)