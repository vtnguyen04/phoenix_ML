from pathlib import Path

import pytest

from src.application.commands.load_model_command import LoadModelCommand
from src.application.handlers.load_model_handler import LoadModelHandler
from src.domain.inference.entities.model import Model
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


class MockArtifactStorage(LocalArtifactStorage):
    async def download(self, remote_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("mock content")
        return local_path

@pytest.mark.asyncio
async def test_load_model_handler_success() -> None:
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    storage = MockArtifactStorage(base_dir=Path("/tmp/test_storage"))
    handler = LoadModelHandler(repo, engine, storage)

    # Setup
    model = Model(id="m1", version="v1", uri="local://m1", framework="onnx")
    await repo.save(model)

    cmd = LoadModelCommand(model_id="m1", model_version="v1")
    
    # Execute
    await handler.execute(cmd)
    
    # Verify loaded
    assert "m1:v1" in engine.loaded_models

@pytest.mark.asyncio
async def test_load_model_not_found() -> None:
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    storage = MockArtifactStorage(base_dir=Path("/tmp/test_storage"))
    handler = LoadModelHandler(repo, engine, storage)

    cmd = LoadModelCommand(model_id="missing", model_version="v1")
    
    with pytest.raises(ValueError, match="Model missing:v1 not found"):
        await handler.execute(cmd)
