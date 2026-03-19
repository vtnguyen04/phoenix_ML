import logging

from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.repositories.drift_report_repository import (
    DriftReportRepository,
)
from src.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from src.domain.monitoring.services.alert_manager import AlertManager
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.infrastructure.monitoring.alert_notifier import AlertNotifier
from src.infrastructure.monitoring.prometheus_metrics import (
    DRIFT_DETECTED_COUNT,
    DRIFT_SCORE,
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Application Service for Monitoring and Self-Healing.
    Responsible for Drift Detection, Alerting, Rollback, and Retraining Triggers.
    """

    MIN_DATA_POINTS = 5

    def __init__(  # noqa: PLR0913
        self,
        log_repo: PredictionLogRepository,
        drift_calculator: DriftCalculator,
        drift_report_repo: DriftReportRepository,
        alert_manager: AlertManager | None = None,
        alert_notifier: AlertNotifier | None = None,
        model_repo: ModelRepository | None = None,
    ) -> None:
        self._log_repo = log_repo
        self._drift_calculator = drift_calculator
        self._drift_report_repo = drift_report_repo
        self._alert_manager = alert_manager
        self._alert_notifier = alert_notifier
        self._model_repo = model_repo

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

        DRIFT_SCORE.labels(
            model_id=model_id, feature_name=feature_name, method=report.method
        ).set(report.statistic)

        if report.drift_detected:
            DRIFT_DETECTED_COUNT.labels(
                model_id=model_id, feature_name=feature_name
            ).inc()

            logger.warning("🚨 DRIFT DETECTED: %s", report.recommendation)

            # 1. Alert Manager — evaluate rules & fire alerts
            await self._dispatch_alerts(model_id, report)

            # 2. Rollback — restore champion model immediately
            await self._rollback_to_champion(model_id)

            # 3. Airflow — trigger retraining pipeline
            await self._trigger_retrain(model_id, report)

        return report

    async def _dispatch_alerts(
        self, model_id: str, report: DriftReport
    ) -> None:
        """Evaluate alert rules and send webhook notifications."""
        if not self._alert_manager:
            return

        alerts = self._alert_manager.evaluate(
            metric_name="drift_score",
            value=report.statistic,
            model_id=model_id,
        )

        if self._alert_notifier and alerts:
            for alert in alerts:
                await self._alert_notifier.notify(alert)
            logger.info(
                "📢 Dispatched %d alert(s) for model %s",
                len(alerts),
                model_id,
            )

    async def _rollback_to_champion(self, model_id: str) -> None:
        """
        Immediately rollback any active challenger to 'archived' stage,
        ensuring the champion model continues serving traffic.
        """
        if not self._model_repo:
            return

        try:
            models = await self._model_repo.get_active_versions(model_id)
            champion = None
            challengers = []

            for m in models:
                role = m.metadata.get("role", "")
                if role == "champion":
                    champion = m
                elif role == "challenger":
                    challengers.append(m)

            if not champion:
                logger.warning(
                    "⚠️ No champion found for %s — skipping rollback",
                    model_id,
                )
                return

            if not challengers:
                logger.info(
                    "ℹ️ No active challengers for %s — nothing to rollback",
                    model_id,
                )
                return

            for c in challengers:
                await self._model_repo.update_stage(
                    model_id, c.version, "archived"
                )
                logger.info(
                    "⏪ ROLLBACK: challenger %s → archived, "
                    "champion %s remains active",
                    c.version,
                    champion.version,
                )
        except Exception as e:
            logger.error("❌ Rollback failed: %s", e)

    async def _trigger_retrain(
        self, model_id: str, report: DriftReport
    ) -> None:
        """
        Trigger the auto-retraining pipeline via Airflow REST API.
        """
        import os  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        logger.info(
            "🔄 Triggering Airflow Retrain DAG for model %s...", model_id
        )

        airflow_url = os.environ.get(
            "AIRFLOW_API_URL", "http://airflow-webserver:8080"
        )

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{airflow_url}/api/v1/dags/retrain_pipeline/dagRuns",
                    json={
                        "conf": {
                            "model_id": model_id,
                            "reason": report.recommendation,
                        }
                    },
                    auth=("admin", "admin"),
                )
                if resp.status_code == 200:  # noqa: PLR2004
                    dag_run_id = resp.json().get("dag_run_id")
                    logger.info(
                        "✅ Airflow DAG triggered: %s", dag_run_id
                    )
                else:
                    logger.error(
                        "❌ Airflow trigger failed: %s - %s",
                        resp.status_code,
                        resp.text,
                    )
        except Exception as e:
            logger.error("❌ Airflow connection error: %s", e)
