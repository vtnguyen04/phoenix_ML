"""Tests for InMemoryPredictionLogRepository."""

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.value_objects.confidence_score import ConfidenceScore
from phoenix_ml.infrastructure.monitoring.in_memory_log_repo import (
    InMemoryPredictionLogRepository,
)


async def test_log_and_get_recent() -> None:
    repo = InMemoryPredictionLogRepository(max_size=100)
    cmd = PredictCommand(model_id="m1", model_version="v1", features=[1.0])
    pred = Prediction(
        model_id="m1",
        model_version="v1",
        result=1,
        confidence=ConfidenceScore(value=0.9),
        latency_ms=5.0,
    )
    await repo.log(cmd, pred)
    logs = await repo.get_recent_logs("m1")
    assert len(logs) == 1
    assert logs[0][1].model_id == "m1"


async def test_get_recent_empty() -> None:
    repo = InMemoryPredictionLogRepository()
    logs = await repo.get_recent_logs("nonexistent")
    assert logs == []


async def test_log_respects_max_size() -> None:
    repo = InMemoryPredictionLogRepository(max_size=2)
    for i in range(5):
        cmd = PredictCommand(model_id="m1", model_version="v1", features=[float(i)])
        pred = Prediction(
            model_id="m1",
            model_version="v1",
            result=i,
            confidence=ConfidenceScore(value=0.5),
            latency_ms=1.0,
        )
        await repo.log(cmd, pred)

    logs = await repo.get_recent_logs("m1")
    assert len(logs) == 2  # max_size=2
