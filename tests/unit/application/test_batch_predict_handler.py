"""Unit tests for batch prediction handler and command."""

import pytest

from src.application.commands.batch_predict_command import BatchPredictCommand
from src.application.handlers.batch_predict_handler import BatchPredictHandler
from src.application.handlers.predict_handler import PredictHandler


async def test_batch_predict_empty_batch() -> None:
    """Empty batch should return empty results with zero latency."""
    # We don't need a real handler for empty batch — mock it
    from unittest.mock import AsyncMock, MagicMock

    mock_handler = MagicMock(spec=PredictHandler)
    mock_handler.execute = AsyncMock(return_value=None)

    batch_handler = BatchPredictHandler(mock_handler)
    command = BatchPredictCommand(model_id="any", batch=[])

    result = await batch_handler.handle(command)

    assert result["total"] == 0
    assert result["successful"] == 0
    assert result["errors"] == []
    assert isinstance(result["batch_latency_ms"], float)


async def test_batch_predict_command_is_frozen() -> None:
    """BatchPredictCommand should be a frozen dataclass with correct fields."""
    cmd = BatchPredictCommand(
        model_id="test",
        batch=[[1.0, 2.0]],
        model_version="v1",
        entity_ids=["e1"],
    )
    assert cmd.model_id == "test"
    assert cmd.batch == [[1.0, 2.0]]
    assert cmd.model_version == "v1"
    assert cmd.entity_ids == ["e1"]

    with pytest.raises(AttributeError):
        cmd.model_id = "changed"  # type: ignore[misc]


async def test_batch_predict_command_defaults() -> None:
    """BatchPredictCommand should have sensible defaults."""
    cmd = BatchPredictCommand(model_id="m1", batch=[[1.0]])
    assert cmd.model_version is None
    assert cmd.entity_ids is None


async def test_batch_predict_processes_all_items() -> None:
    """Batch handler should attempt to process every item in the batch."""
    from unittest.mock import AsyncMock, MagicMock

    from src.domain.inference.entities.prediction import Prediction
    from src.domain.inference.value_objects.confidence_score import ConfidenceScore

    mock_prediction = Prediction(
        model_id="m1",
        model_version="v1",
        result=1,
        confidence=ConfidenceScore(value=0.9),
        latency_ms=1.0,
    )

    mock_handler = MagicMock(spec=PredictHandler)
    mock_handler.execute = AsyncMock(return_value=mock_prediction)

    batch_handler = BatchPredictHandler(mock_handler)
    command = BatchPredictCommand(
        model_id="m1",
        batch=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        model_version="v1",
    )

    result = await batch_handler.handle(command)

    assert result["total"] == 3
    assert result["successful"] == 3
    assert len(result["errors"]) == 0
    assert mock_handler.execute.call_count == 3


async def test_batch_predict_handles_partial_failure() -> None:
    """Batch handler should gracefully handle individual prediction failures."""
    from unittest.mock import AsyncMock, MagicMock

    from src.domain.inference.entities.prediction import Prediction
    from src.domain.inference.value_objects.confidence_score import ConfidenceScore

    ok = Prediction(
        model_id="m1",
        model_version="v1",
        result=0,
        confidence=ConfidenceScore(value=0.8),
        latency_ms=1.0,
    )

    call_count = 0

    async def _side_effect(*args: object, **kwargs: object) -> Prediction:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            msg = "model error"
            raise ValueError(msg)
        return ok

    mock_handler = MagicMock(spec=PredictHandler)
    mock_handler.execute = AsyncMock(side_effect=_side_effect)

    batch_handler = BatchPredictHandler(mock_handler)
    command = BatchPredictCommand(model_id="m1", batch=[[1.0], [2.0], [3.0]])

    result = await batch_handler.handle(command)

    assert result["total"] == 3
    assert result["successful"] == 2
    assert len(result["errors"]) == 1
    assert result["errors"][0]["index"] == 1
