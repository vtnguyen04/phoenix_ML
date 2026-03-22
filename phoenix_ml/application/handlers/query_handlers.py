"""
CQRS Read-Side Query Handlers.

Each handler receives a query object and returns data from the appropriate
repository without side effects.
"""

from typing import Any

from phoenix_ml.application.queries import (
    GetDriftReportQuery,
    GetModelPerformanceQuery,
    GetModelQuery,
    GetPredictionLogsQuery,
)
from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.model_registry.repositories.model_repository import ModelRepository
from phoenix_ml.domain.monitoring.entities.drift_report import DriftReport
from phoenix_ml.domain.monitoring.repositories.drift_report_repository import (
    DriftReportRepository,
)
from phoenix_ml.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from phoenix_ml.domain.monitoring.services.model_evaluator import IModelEvaluator


class GetModelQueryHandler:
    """Retrieves model information from the registry."""

    def __init__(self, model_repo: ModelRepository) -> None:
        self._model_repo = model_repo

    async def execute(self, query: GetModelQuery) -> Model | None:
        if query.version:
            return await self._model_repo.get_by_id(query.model_id, query.version)
        return await self._model_repo.get_champion(query.model_id)


class GetDriftReportQueryHandler:
    """Retrieves historical drift reports for a model."""

    def __init__(self, drift_repo: DriftReportRepository) -> None:
        self._drift_repo = drift_repo

    async def execute(self, query: GetDriftReportQuery) -> list[DriftReport]:
        return await self._drift_repo.get_history(query.model_id, query.limit)


class GetPredictionLogsQueryHandler:
    """Retrieves recent prediction logs for auditing and analysis."""

    def __init__(self, log_repo: PredictionLogRepository) -> None:
        self._log_repo = log_repo

    async def execute(self, query: GetPredictionLogsQuery) -> list[dict[str, Any]]:
        raw_logs = await self._log_repo.get_recent_logs(query.model_id, query.limit)
        return [
            {
                "model_id": command.model_id,
                "model_version": command.model_version,
                "result": prediction.result,
                "confidence": prediction.confidence.value,
                "latency_ms": prediction.latency_ms,
            }
            for command, prediction in raw_logs
        ]


class GetModelPerformanceQueryHandler:
    """Computes model performance metrics from logged predictions."""

    def __init__(
        self,
        log_repo: PredictionLogRepository,
        evaluator: IModelEvaluator,
    ) -> None:
        self._log_repo = log_repo
        self._evaluator = evaluator

    async def execute(self, query: GetModelPerformanceQuery) -> dict[str, Any]:
        logs = await self._log_repo.get_recent_logs(query.model_id, limit=1000)
        if not logs:
            return {
                "model_id": query.model_id,
                "version": query.version,
                "total_predictions": 0,
                "metrics": {},
            }

        total = len(logs)
        avg_latency = sum(p.latency_ms for _, p in logs) / total
        avg_confidence = sum(p.confidence.value for _, p in logs) / total

        return {
            "model_id": query.model_id,
            "version": query.version,
            "total_predictions": total,
            "metrics": {
                "avg_latency_ms": round(avg_latency, 2),
                "avg_confidence": round(avg_confidence, 4),
            },
        }
