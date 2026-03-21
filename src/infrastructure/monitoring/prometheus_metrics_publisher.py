"""
PrometheusMetricsPublisher — Infrastructure adapter for MetricsPublisher.

Implements the domain port using prometheus_client gauges/counters/histograms.
Follows Open/Closed Principle: add new metric types via _METRIC_MAP without
modifying existing code.

To swap to another backend, create a new adapter (e.g. DatadogMetricsPublisher)
and change one line in container.py.
"""

import logging
from typing import Any

from src.domain.monitoring.services.metrics_publisher import MetricsPublisher
from src.infrastructure.monitoring.prometheus_metrics import (
    DRIFT_DETECTED_COUNT,
    DRIFT_SCORE,
    INFERENCE_LATENCY,
    MODEL_ACCURACY,
    MODEL_CONFIDENCE,
    MODEL_F1_SCORE,
    MODEL_MAE,
    MODEL_PRIMARY_METRIC,
    MODEL_R2,
    MODEL_RMSE,
    PREDICTION_COUNT,
)

logger = logging.getLogger(__name__)

# ── Metric mapping: metric_name → Prometheus gauge (OCP) ──────────
_METRIC_MAP: dict[str, Any] = {
    # Classification
    "accuracy": MODEL_ACCURACY,
    "f1_score": MODEL_F1_SCORE,
    "f1_macro": MODEL_F1_SCORE,  # alias for image-class
    "test_accuracy": MODEL_ACCURACY,  # alias from some training scripts
    # Regression
    "rmse": MODEL_RMSE,
    "mae": MODEL_MAE,
    "r2": MODEL_R2,
}

# ── Primary metric per task type (extensible) ─────────────────────
_PRIMARY_METRIC_BY_TASK: dict[str, str] = {
    "classification": "accuracy",
    "regression": "rmse",
}


class PrometheusMetricsPublisher(MetricsPublisher):
    """Publishes ALL observability metrics to Prometheus.

    Covers prediction counts, latency, confidence, model evaluation,
    and drift detection — all via the abstract MetricsPublisher interface.
    """

    # ── Prediction observability ──────────────────────────────────

    def record_prediction(
        self,
        model_id: str,
        version: str,
        status: str,
    ) -> None:
        PREDICTION_COUNT.labels(
            model_id=model_id,
            version=version,
            status=status,
        ).inc()

    def record_latency(
        self,
        model_id: str,
        version: str,
        latency_seconds: float,
    ) -> None:
        INFERENCE_LATENCY.labels(
            model_id=model_id,
            version=version,
        ).observe(latency_seconds)

    def record_confidence(
        self,
        model_id: str,
        version: str,
        confidence: float,
    ) -> None:
        MODEL_CONFIDENCE.labels(
            model_id=model_id,
            version=version,
        ).observe(confidence)

    # ── Model evaluation metrics ──────────────────────────────────

    def publish_model_metrics(
        self,
        model_id: str,
        version: str,
        metrics: dict[str, Any],
    ) -> None:
        task_type = metrics.get("task_type", "classification")

        # Set all matching gauges via _METRIC_MAP (OCP)
        for metric_name, gauge in _METRIC_MAP.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    gauge.labels(model_id=model_id, version=version).set(value)

        # Set task-agnostic primary metric
        primary = _PRIMARY_METRIC_BY_TASK.get(task_type, "accuracy")
        if primary in metrics:
            MODEL_PRIMARY_METRIC.labels(
                model_id=model_id,
                version=version,
                metric_name=primary,
            ).set(metrics[primary])

        logger.info(
            "📊 Prometheus gauges set for %s:%s (task=%s)",
            model_id,
            version,
            task_type,
        )

    # ── Drift detection ───────────────────────────────────────────

    def publish_drift_score(
        self,
        model_id: str,
        feature_name: str,
        method: str,
        score: float,
    ) -> None:
        DRIFT_SCORE.labels(
            model_id=model_id,
            feature_name=feature_name,
            method=method,
        ).set(score)

    def record_drift_detected(
        self,
        model_id: str,
        feature_name: str,
    ) -> None:
        DRIFT_DETECTED_COUNT.labels(
            model_id=model_id,
            feature_name=feature_name,
        ).inc()
