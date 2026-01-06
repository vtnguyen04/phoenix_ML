from pathlib import Path

import pytest

from src.application.dto.prediction_request import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.entities.model import Model
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


class MockArtifactStorage(LocalArtifactStorage):
    async def download(self, remote_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("mock content")
        return local_path

@pytest.fixture
def predict_handler():
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    fs = InMemoryFeatureStore()
    storage = MockArtifactStorage(base_dir=Path("/tmp/test_storage"))
    return PredictHandler(repo, engine, fs, storage), repo, fs

@pytest.mark.asyncio
async def test_predict_handler_with_direct_features(predict_handler):
    handler, repo, _ = predict_handler
    
    # Register model
    model = Model(id="m1", version="v1", uri="local://tmp/m1", framework="onnx")
    await repo.save(model)
    
    # Execute with explicit features
    command = PredictCommand(
        model_id="m1", model_version="v1", features=[1.0, 2.0]
    )
    prediction = await handler.execute(command)
    assert prediction.result == pytest.approx(1.5)  # noqa: PLR2004

@pytest.mark.asyncio
async def test_predict_handler_with_feature_store(predict_handler):
    handler, repo, fs = predict_handler
    
    # Register model & Seed features
    model = Model(id="m2", version="v1", uri="local://tmp/m2", framework="onnx")
    await repo.save(model)
    fs.add_features("user-1", {"f1": 10.0, "f2": 20.0, "f3": 30.0})
    
    # Execute with entity_id
    command = PredictCommand(
        model_id="m2", model_version="v1", entity_id="user-1"
    )
    prediction = await handler.execute(command)
    
    assert prediction.result == pytest.approx(20.0)  # noqa: PLR2004

@pytest.mark.asyncio
async def test_predict_handler_missing_features_raises_error(predict_handler):
    handler, repo, _ = predict_handler
    
    model = Model(id="m3", version="v1", uri="local://tmp/m3", framework="onnx")
    await repo.save(model)
    
    command = PredictCommand(model_id="m3", model_version="v1")
    
    with pytest.raises(ValueError, match="No features provided"):
        await handler.execute(command)
