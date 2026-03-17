from abc import ABC, abstractmethod

from src.domain.monitoring.entities.drift_report import DriftReport


class DriftReportRepository(ABC):
    @abstractmethod
    async def save(self, model_id: str, report: DriftReport) -> None:
        pass

    @abstractmethod
    async def get_history(self, model_id: str, limit: int = 100) -> list[DriftReport]:
        pass
