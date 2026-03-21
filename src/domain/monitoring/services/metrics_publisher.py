"""
MetricsPublisher — Domain Port for ALL observability metrics.

Follows Dependency Inversion Principle (DIP): the application layer
defines WHAT to observe, infrastructure decides HOW (Prometheus,
Datadog, CloudWatch, StatsD, etc.).

This single port replaces ALL direct Prometheus imports from the
application layer.  To switch backends, swap the adapter in container.py:

    metrics_publisher: MetricsPublisher = DatadogMetricsPublisher()
"""

from abc import ABC, abstractmethod
from typing import Any


class MetricsPublisher(ABC):
    """Abstract port for ALL observability metrics.

    Covers:
    - Prediction recording (count, latency, confidence)
    - Model evaluation metrics (accuracy, RMSE, F1, …)
    - Drift detection scores
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
