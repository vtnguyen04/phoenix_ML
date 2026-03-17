import logging

from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.infrastructure.monitoring.prometheus_metrics import (
    DRIFT_DETECTED_COUNT,
    DRIFT_SCORE,
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Application Service for Monitoring and Self-Healing.
    Responsible for Drift Detection and Retraining Triggers.
    """

    MIN_DATA_POINTS = 5  # Lower threshold for demo purposes

    def __init__(
        self, log_repo: PredictionLogRepository, drift_calculator: DriftCalculator
    ) -> None:
        self._log_repo = log_repo
        self._drift_calculator = drift_calculator

    async def check_drift(
        self,
        model_id: str,
        reference_data: list[float],
        feature_index: int = 0,
        test_type: str = "ks",
    ) -> DriftReport:
        """
        Check for drift on a specific feature index against reference data.
        """
        # 1. Get recent production logs
        logs = await self._log_repo.get_recent_logs(model_id, limit=1000)

        if not logs:
            raise ValueError(f"No logs found for model {model_id}")

        # 2. Extract feature values
        current_data: list[float] = []
        for cmd, _ in logs:
            if cmd.features:
                try:
                    current_data.append(cmd.features[feature_index])
                except IndexError:
                    continue

        if len(current_data) < self.MIN_DATA_POINTS:
            raise ValueError(
                f"Not enough data points ({len(current_data)}) to calculate drift"
            )

        # 3. Calculate Drift
        feature_name = f"feature_{feature_index}"
        report = self._drift_calculator.calculate_drift(
            feature_name=feature_name,
            reference_data=reference_data,
            current_data=current_data,
            test_type=test_type,
        )

        # 4. Update Prometheus Metrics
        DRIFT_SCORE.labels(
            model_id=model_id, feature_name=feature_name, method=report.method
        ).set(report.statistic)

        if report.drift_detected:
            DRIFT_DETECTED_COUNT.labels(
                model_id=model_id, feature_name=feature_name
            ).inc()

            logger.warning("🚨 DRIFT DETECTED: %s", report.recommendation)
            await self._trigger_retrain(model_id, report)

        return report

    async def _trigger_retrain(self, model_id: str, report: DriftReport) -> None:
        """
        Trigger the auto-retraining pipeline.
        In a real system, this would send a message to Kafka or call Airflow API.
        """
        logger.info("🔄 Triggering Retrain Handler for model %s...", model_id)
        # Mock triggering a handler
        # In a real implementation:
        # command = TriggerRetrainCommand(model_id=model_id,
        #                                reason=report.recommendation)
        # await self._retrain_handler.execute(command)
        pass
