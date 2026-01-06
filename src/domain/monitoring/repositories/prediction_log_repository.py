from abc import ABC, abstractmethod

from src.application.commands.predict_command import PredictCommand
from src.domain.inference.entities.prediction import Prediction


class PredictionLogRepository(ABC):
    """
    Interface for storing prediction logs for monitoring.
    """
    
    @abstractmethod
    async def log(self, command: PredictCommand, prediction: Prediction) -> None:
        """Log a single prediction event"""
        pass

    @abstractmethod
    async def get_recent_logs(
        self, model_id: str, limit: int = 1000
    ) -> list[tuple[PredictCommand, Prediction]]:
        """Retrieve recent logs for analysis"""
        pass
