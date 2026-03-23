"""Abstract port for observability metrics.

Defines the contract for recording predictions, latency, confidence,
model evaluation results, and drift scores. Infrastructure adapters
(Prometheus, Datadog, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class MetricsPublisher(ABC):
    """Abstract interface for recording observability metrics.

    Methods cover prediction counts, latency, confidence histograms,
    model evaluation results, and drift detection scores.
    """

    # ── Prediction observability ──────────────────────────────────

    @abstractmethod
    def record_prediction(
        self,
        model_id: str,
        version: str,
        status: str,
    ) -> None:
        """Increment the prediction counter.

        Args:
            model_id: Model identifier.
            version: Model version string.
            status: "success" or "error".
        """

    @abstractmethod
    def record_latency(
        self,
        model_id: str,
        version: str,
        latency_seconds: float,
    ) -> None:
        """Record inference latency (seconds)."""

    @abstractmethod
    def record_confidence(
        self,
        model_id: str,
        version: str,
        confidence: float,
    ) -> None:
        """Record prediction confidence score."""

    # ── Model evaluation metrics ──────────────────────────────────

    @abstractmethod
    def publish_model_metrics(
        self,
        model_id: str,
        version: str,
        metrics: dict[str, Any],
    ) -> None:
        """Publish model training/evaluation metrics.

        Implementations should inspect ``metrics["task_type"]`` to
        determine which gauges to set.

        Args:
            model_id: Model identifier.
            version: Model version.
            metrics: Dict of metric_name → value from training output.
        """

    # ── Drift detection ───────────────────────────────────────────

    @abstractmethod
    def publish_drift_score(
        self,
        model_id: str,
        feature_name: str,
        method: str,
        score: float,
    ) -> None:
        """Publish a drift statistic for a specific feature."""

    @abstractmethod
    def record_drift_detected(
        self,
        model_id: str,
        feature_name: str,
    ) -> None:
        """Increment the drift-detected event counter."""
