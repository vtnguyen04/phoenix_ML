from collections import deque
from typing import Any

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)


class InMemoryPredictionLogRepository(PredictionLogRepository):
    """
    In-memory implementation of prediction logger using a circular buffer (deque).
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._logs: dict[str, deque[tuple[PredictCommand, Prediction]]] = {}
        self._max_size = max_size

    async def log(
        self,
        command: PredictCommand,
        prediction: Prediction,
        *,
        prediction_id: str | None = None,
    ) -> None:
        model_id = prediction.model_id
        if model_id not in self._logs:
            self._logs[model_id] = deque(maxlen=self._max_size)

        self._logs[model_id].append((command, prediction))

    async def get_recent_logs(
        self, model_id: str, limit: int = 1000
    ) -> list[tuple[PredictCommand, Prediction]]:
        if model_id not in self._logs:
            return []

        # Convert deque to list and slice
        logs = list(self._logs[model_id])
        return logs[-limit:]

    async def export_labeled_logs(
        self, model_id: str, *, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """In-memory repo does not track ground_truth — returns empty list."""
        return []
