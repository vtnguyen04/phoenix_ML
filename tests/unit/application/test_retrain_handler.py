import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.application.commands.trigger_retrain_command import TriggerRetrainCommand
from src.application.handlers.retrain_handler import RetrainHandler
from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.domain.monitoring.services.model_evaluator import ClassificationEvaluator
from src.domain.shared.event_bus import DomainEventBus


@pytest.fixture
def mock_repo() -> AsyncMock:
    return AsyncMock(spec=ModelRepository)


@pytest.fixture
def mock_evaluator() -> Mock:
    m = Mock(spec=ClassificationEvaluator)
    m.is_better.return_value = True
    return m


@pytest.fixture
def retrain_handler(mock_repo: AsyncMock, mock_evaluator: Mock) -> RetrainHandler:
    return RetrainHandler(
        project_root=Path("/tmp/phoenix"),
        model_repo=mock_repo,
        evaluator=mock_evaluator,
        event_bus=DomainEventBus(),
    )


@pytest.mark.asyncio
async def test_retrain_handler_success(
    retrain_handler: RetrainHandler, mock_repo: AsyncMock
) -> None:
    # Setup
    command = TriggerRetrainCommand(model_id="m1", reason="test drift")

    # Mock subprocess
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"done", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        # Metrics to return
        metrics = {
            "accuracy": 0.9,
            "f1_score": 0.85,
            "precision": 0.8,
            "recall": 0.8,
        }
        metrics_data = json.dumps(metrics)

        # Mock Path and open
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open(read_data=metrics_data)),
        ):
            success = await retrain_handler.execute(command)

            assert success is True
            mock_exec.assert_called_once()
            # Verify model was saved
            mock_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrain_handler_failure(retrain_handler: RetrainHandler) -> None:
    # Setup
    command = TriggerRetrainCommand(model_id="m1", reason="test drift")

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1
        mock_exec.return_value = mock_process

        with patch("pathlib.Path.exists", return_value=True):
            success = await retrain_handler.execute(command)

            assert success is False
