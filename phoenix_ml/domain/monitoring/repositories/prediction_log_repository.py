from abc import ABC, abstractmethod
from typing import Any

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.domain.inference.entities.prediction import Prediction


class PredictionLogRepository(ABC):
    """Abstract repository for persisting prediction logs."""

    @abstractmethod
    async def log(
        self,
        command: PredictCommand,
        prediction: Prediction,
        *,
        prediction_id: str | None = None,
    ) -> None:
        """Log a single prediction event"""
        pass

    @abstractmethod
    async def get_recent_logs(
        self, model_id: str, limit: int = 1000
    ) -> list[tuple[PredictCommand, Prediction]]:
        """Retrieve recent logs for analysis"""
        pass

    @abstractmethod
    async def export_labeled_logs(
        self, model_id: str, *, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Export prediction logs that have ground_truth labels.

        Returns flat dicts with features + ground_truth for training.
        Only returns logs where ground_truth is NOT NULL.
        """
        pass
