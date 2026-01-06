from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from src.domain.monitoring.services.drift_calculator import DriftCalculator


class MonitoringService:
    """
    Application Service for Monitoring use cases.
    """
    MIN_DATA_POINTS = 10
    
    def __init__(
        self,
        log_repo: PredictionLogRepository,
        drift_calculator: DriftCalculator
    ) -> None:
        self._log_repo = log_repo
        self._drift_calculator = drift_calculator

    async def check_drift(
        self, 
        model_id: str, 
        reference_data: list[float],
        feature_index: int = 0
    ) -> DriftReport:
        """
        Check for drift on a specific feature index against provided reference data.
        """
        # 1. Get recent production logs
        logs = await self._log_repo.get_recent_logs(model_id, limit=1000)
        
        if not logs:
            raise ValueError(f"No logs found for model {model_id}")

        # 2. Extract feature values (assuming feature vector is flat list)
        # We need to handle the case where features might be None in command 
        # (fetched from store). But for drift calculation, we really need the 
        # resolved features used for inference.
        
        current_data: list[float] = []
        for cmd, _ in logs:
            if cmd.features:
                try:
                    current_data.append(cmd.features[feature_index])
                except IndexError:
                    continue
        
        if len(current_data) < self.MIN_DATA_POINTS:
             raise ValueError("Not enough data points to calculate drift")

        # 3. Calculate Drift
        report = self._drift_calculator.calculate_drift(
            feature_name=f"feature_{feature_index}",
            reference_data=reference_data,
            current_data=current_data
        )
        
        return report
