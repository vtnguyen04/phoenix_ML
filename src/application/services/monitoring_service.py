import logging

from src.config import get_settings
from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.repositories.drift_report_repository import (
    DriftReportRepository,
)
from src.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from src.domain.monitoring.services.alert_manager import AlertManager
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.domain.shared.domain_events import DriftDetected, DriftScorePublished
from src.domain.shared.event_bus import DomainEventBus

logger = logging.getLogger(__name__)
_settings = get_settings()


class MonitoringService:
    """
    Application Service for Monitoring — Detection Only.

    Detects drift and triggers the Airflow Self-Healing Pipeline.
    Uses Observer Pattern: emits DriftScorePublished / DriftDetected events.
    """

    MIN_DATA_POINTS = _settings.MONITORING_MIN_DATA_POINTS

    def __init__(
        self,
        log_repo: PredictionLogRepository,
        drift_calculator: DriftCalculator,
        drift_report_repo: DriftReportRepository,
        alert_manager: AlertManager | None = None,
        event_bus: DomainEventBus | None = None,
    ) -> None:
        self._log_repo = log_repo
        self._drift_calculator = drift_calculator
        self._drift_report_repo = drift_report_repo
        self._alert_manager = alert_manager
        self._event_bus = event_bus

    async def check_drift(
        self,
        model_id: str,
        reference_data: list[float],
        feature_index: int = 0,
        test_type: str = "ks",
    ) -> DriftReport:
        """
        Check for drift on a specific feature index against reference data.
        If drift is detected, trigger the Airflow self-healing pipeline
        (with deduplication — skips if a run is already active).
        """
        logs = await self._log_repo.get_recent_logs(model_id, limit=1000)

        if not logs:
            raise ValueError(f"No logs found for model {model_id}")

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

        feature_name = f"feature_{feature_index}"
        report = self._drift_calculator.calculate_drift(
            feature_name=feature_name,
            reference_data=reference_data,
            current_data=current_data,
            test_type=test_type,
        )

        await self._drift_report_repo.save(model_id, report)

        # Emit domain events (Observer Pattern)
        if self._event_bus:
            self._event_bus.publish(DriftScorePublished(
                model_id=model_id,
                feature_name=feature_name,
                method=report.method,
                score=report.statistic,
            ))

            if report.drift_detected:
                self._event_bus.publish(DriftDetected(
                    model_id=model_id,
                    feature_name=feature_name,
                    score=report.statistic,
                    method=report.method,
                ))

        if report.drift_detected:
            logger.warning("🚨 DRIFT DETECTED: %s", report.recommendation)

            if self._alert_manager:
                self._alert_manager.evaluate(
                    metric_name="drift_score",
                    value=report.statistic,
                    model_id=model_id,
                )

            await self._trigger_self_healing(model_id, report)

        return report

    async def _trigger_self_healing(
        self, model_id: str, report: DriftReport
    ) -> None:
        """
        Trigger Airflow self-healing pipeline via REST API.
        Checks if a run is already active before triggering a new one.
        """
        import httpx  # noqa: PLC0415

        airflow_url = _settings.AIRFLOW_API_URL
        airflow_user = _settings.AIRFLOW_ADMIN_USER
        airflow_pass = _settings.AIRFLOW_ADMIN_PASSWORD
        dag_id = _settings.AIRFLOW_DAG_ID

        try:
            async with httpx.AsyncClient(
                base_url=airflow_url,
                auth=(airflow_user, airflow_pass),
                timeout=10.0,
            ) as client:
                # Check for already-running instances
                runs_resp = await client.get(
                    f"/dags/{dag_id}/dagRuns",
                    params={"state": "running", "limit": 1},
                )
                if runs_resp.status_code == 200:  # noqa: PLR2004
                    active_runs = runs_resp.json().get("dag_runs", [])
                    if active_runs:
                        logger.info(
                            "⏳ Self-healing pipeline already running for %s — skipping",
                            model_id,
                        )
                        return

                # Trigger new DAG run
                trigger_resp = await client.post(
                    f"/dags/{dag_id}/dagRuns",
                    json={
                        "conf": {
                            "model_id": model_id,
                            "drift_score": report.statistic,
                            "drift_method": report.method,
                            "feature_name": report.feature_name,
                        }
                    },
                )

                if trigger_resp.status_code in (200, 409):
                    logger.info(
                        "✅ Self-healing pipeline triggered for %s", model_id
                    )
                else:
                    logger.error(
                        "❌ Failed to trigger self-healing: %s %s",
                        trigger_resp.status_code,
                        trigger_resp.text,
                    )

        except Exception as e:
            logger.error("❌ Failed to connect to Airflow: %s", e)
