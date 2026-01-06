from collections import deque

from src.application.commands.predict_command import PredictCommand
from src.domain.inference.entities.prediction import Prediction
from src.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)


class InMemoryPredictionLogRepository(PredictionLogRepository):
    """
    In-memory implementation of prediction logger using a circular buffer (deque).
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._logs: dict[str, deque[tuple[PredictCommand, Prediction]]] = {}
        self._max_size = max_size

    async def log(self, command: PredictCommand, prediction: Prediction) -> None:
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
