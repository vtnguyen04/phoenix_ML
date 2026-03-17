from unittest.mock import AsyncMock

import pytest

from src.application.commands.load_model_command import LoadModelCommand
from src.application.handlers.load_model_handler import LoadModelHandler
from src.domain.inference.entities.model import Model
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.model_registry.repositories.model_repository import ModelRepository


@pytest.fixture
def mock_repo() -> AsyncMock:
    return AsyncMock(spec=ModelRepository)


@pytest.fixture
def mock_engine() -> AsyncMock:
    return AsyncMock(spec=InferenceEngine)


@pytest.mark.asyncio
async def test_load_model_handler_success(
    mock_repo: AsyncMock, mock_engine: AsyncMock
) -> None:
    handler = LoadModelHandler(mock_repo, mock_engine)

    # Setup
    model = Model(id="m1", version="v1", uri="loc://v1", framework="onnx")
    mock_repo.get_by_id.return_value = model

    command = LoadModelCommand(model_id="m1", model_version="v1")

    # Execute
    success = await handler.execute(command)

    # Verify
    assert success is True
    mock_repo.get_by_id.assert_called_once_with("m1", "v1")
    mock_engine.load.assert_called_once_with(model)


@pytest.mark.asyncio
async def test_load_model_handler_not_found(
    mock_repo: AsyncMock, mock_engine: AsyncMock
) -> None:
    handler = LoadModelHandler(mock_repo, mock_engine)

    # Setup
    mock_repo.get_by_id.return_value = None

    command = LoadModelCommand(model_id="m1", model_version="v1")

    # Execute & Verify
    with pytest.raises(ValueError, match="not found"):
        await handler.execute(command)
