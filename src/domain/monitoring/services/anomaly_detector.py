from dataclasses import dataclass
from enum import Enum

import numpy as np

_EPSILON = 1e-10
_ANOMALY_RATIO_THRESHOLD = 0.1


class AnomalyType(Enum):
    PREDICTION_ANOMALY = "prediction_anomaly"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE = "error_rate"


@dataclass(frozen=True)
class AnomalyReport:
    anomaly_type: AnomalyType
    is_anomalous: bool
    score: float
    threshold: float
    detail: str
    sample_size: int


class AnomalyDetector:
    def __init__(
        self,
        z_score_threshold: float = 3.0,
        latency_multiplier: float = 2.0,
        error_rate_threshold: float = 0.05,
    ) -> None:
        self._z_threshold = z_score_threshold
        self._latency_multiplier = latency_multiplier
        self._error_rate_threshold = error_rate_threshold

    def detect_prediction_anomaly(
        self,
        confidence_scores: list[float],
        baseline_mean: float | None = None,
        baseline_std: float | None = None,
    ) -> AnomalyReport:
        arr = np.array(confidence_scores, dtype=np.float64)
        if len(arr) == 0:
            return AnomalyReport(
                anomaly_type=AnomalyType.PREDICTION_ANOMALY,
                is_anomalous=False,
                score=0.0,
                threshold=self._z_threshold,
                detail="No data to analyze",
                sample_size=0,
            )

        mean = baseline_mean if baseline_mean is not None else float(np.mean(arr))
        std = baseline_std if baseline_std is not None else float(np.std(arr))

        if std < _EPSILON:
            return AnomalyReport(
                anomaly_type=AnomalyType.PREDICTION_ANOMALY,
                is_anomalous=False,
                score=0.0,
                threshold=self._z_threshold,
                detail="Zero variance in confidence scores",
                sample_size=len(arr),
            )

        z_scores = np.abs((arr - mean) / std)
        max_z = float(np.max(z_scores))
        anomaly_ratio = float(np.mean(z_scores > self._z_threshold))
        is_anomalous = anomaly_ratio > _ANOMALY_RATIO_THRESHOLD

        return AnomalyReport(
            anomaly_type=AnomalyType.PREDICTION_ANOMALY,
            is_anomalous=is_anomalous,
            score=max_z,
            threshold=self._z_threshold,
            detail=f"{anomaly_ratio:.1%} of predictions exceed z-score threshold ({max_z:.2f} max)",
            sample_size=len(arr),
        )

    def detect_latency_spike(
        self,
        latencies_ms: list[float],
        baseline_p99_ms: float,
    ) -> AnomalyReport:
        arr = np.array(latencies_ms, dtype=np.float64)
        if len(arr) == 0:
            return AnomalyReport(
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                is_anomalous=False,
                score=0.0,
                threshold=baseline_p99_ms * self._latency_multiplier,
                detail="No latency data",
                sample_size=0,
            )

        current_p99 = float(np.percentile(arr, 99))
        spike_threshold = baseline_p99_ms * self._latency_multiplier
        ratio = current_p99 / baseline_p99_ms if baseline_p99_ms > 0 else 0.0
        is_spike = current_p99 > spike_threshold

        return AnomalyReport(
            anomaly_type=AnomalyType.LATENCY_SPIKE,
            is_anomalous=is_spike,
            score=current_p99,
            threshold=spike_threshold,
            detail=f"p99={current_p99:.1f}ms ({ratio:.1f}x baseline {baseline_p99_ms:.1f}ms)",
            sample_size=len(arr),
        )

    def detect_error_rate(
        self,
        total_requests: int,
        error_count: int,
    ) -> AnomalyReport:
        if total_requests == 0:
            return AnomalyReport(
                anomaly_type=AnomalyType.ERROR_RATE,
                is_anomalous=False,
                score=0.0,
                threshold=self._error_rate_threshold,
                detail="No requests recorded",
                sample_size=0,
            )

        rate = error_count / total_requests
        is_high = rate > self._error_rate_threshold

        return AnomalyReport(
            anomaly_type=AnomalyType.ERROR_RATE,
            is_anomalous=is_high,
            score=rate,
            threshold=self._error_rate_threshold,
            detail=f"{error_count}/{total_requests} errors ({rate:.2%})",
            sample_size=total_requests,
        )
