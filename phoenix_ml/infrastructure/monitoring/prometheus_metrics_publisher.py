"""Prometheus implementation of ``MetricsPublisher``."""

from __future__ import annotations

import logging
from typing import Any

from phoenix_ml.domain.monitoring.services.metrics_publisher import MetricsPublisher
from phoenix_ml.infrastructure.monitoring.prometheus_metrics import (
    DRIFT_DETECTED_COUNT,
    DRIFT_SCORE,
    INFERENCE_LATENCY,
    MODEL_CONFIDENCE,
    MODEL_PRIMARY_METRIC,
    PREDICTION_COUNT,
    PREDICTION_ERROR_RATE,
)

logger = logging.getLogger(__name__)


class PrometheusMetricsPublisher(MetricsPublisher):

    def record_prediction(
        self,
        model_id: str,
        version: str,
        status: str = "ok",
    ) -> None:
        PREDICTION_COUNT.labels(
            model_id=model_id, version=version, status=status
        ).inc()

    def record_latency(
        self,
        model_id: str,
        version: str,
        latency_seconds: float,
    ) -> None:
        INFERENCE_LATENCY.labels(
            model_id=model_id, version=version
        ).observe(latency_seconds)

    def record_confidence(
        self,
        model_id: str,
        version: str,
        confidence: float,
    ) -> None:
        MODEL_CONFIDENCE.labels(
            model_id=model_id, version=version
        ).observe(confidence)

    def publish_model_metrics(
        self,
        model_id: str,
        version: str,
        metrics: dict[str, Any],
        task_type: str = "classification",
    ) -> None:
        primary_key = "accuracy" if task_type == "classification" else "rmse"
        value = metrics.get(primary_key)
        if value is not None and isinstance(value, (int, float)):
            MODEL_PRIMARY_METRIC.labels(
                model_id=model_id,
                version=version,
                metric_name=primary_key,
            ).set(float(value))

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

    def update_error_rate(
        self,
        model_id: str,
        error_rate: float,
    ) -> None:
        PREDICTION_ERROR_RATE.labels(model_id=model_id).set(error_rate)
