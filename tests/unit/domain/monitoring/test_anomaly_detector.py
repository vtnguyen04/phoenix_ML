import numpy as np
import pytest

from src.domain.monitoring.services.anomaly_detector import (
    AnomalyDetector,
    AnomalyType,
)


@pytest.fixture()
def detector() -> AnomalyDetector:
    return AnomalyDetector(z_score_threshold=3.0, latency_multiplier=2.0, error_rate_threshold=0.05)


class TestPredictionAnomaly:
    def test_normal_distribution_no_anomaly(self, detector: AnomalyDetector) -> None:
        scores = np.random.normal(0.8, 0.05, 100).tolist()
        report = detector.detect_prediction_anomaly(scores)
        assert report.anomaly_type == AnomalyType.PREDICTION_ANOMALY
        assert not report.is_anomalous
        assert report.sample_size == 100

    def test_with_anomalous_values(self, detector: AnomalyDetector) -> None:
        normal = [0.8] * 80
        anomalous = [0.1] * 20
        report = detector.detect_prediction_anomaly(
            normal + anomalous, baseline_mean=0.8, baseline_std=0.02
        )
        assert report.is_anomalous
        assert report.score > 3.0

    def test_empty_data(self, detector: AnomalyDetector) -> None:
        report = detector.detect_prediction_anomaly([])
        assert not report.is_anomalous
        assert report.sample_size == 0

    def test_zero_variance(self, detector: AnomalyDetector) -> None:
        report = detector.detect_prediction_anomaly([0.5, 0.5, 0.5])
        assert not report.is_anomalous
        assert report.detail == "Zero variance in confidence scores"

    def test_with_baseline(self, detector: AnomalyDetector) -> None:
        scores = [0.1, 0.12, 0.08, 0.11, 0.09]
        report = detector.detect_prediction_anomaly(
            scores, baseline_mean=0.8, baseline_std=0.05
        )
        assert report.is_anomalous


class TestLatencySpike:
    def test_normal_latency(self, detector: AnomalyDetector) -> None:
        latencies = np.random.normal(20, 3, 100).tolist()
        report = detector.detect_latency_spike(latencies, baseline_p99_ms=30.0)
        assert report.anomaly_type == AnomalyType.LATENCY_SPIKE
        assert not report.is_anomalous

    def test_spike_detected(self, detector: AnomalyDetector) -> None:
        latencies = np.random.normal(80, 10, 100).tolist()
        report = detector.detect_latency_spike(latencies, baseline_p99_ms=30.0)
        assert report.is_anomalous
        assert report.score > 60.0

    def test_empty_latencies(self, detector: AnomalyDetector) -> None:
        report = detector.detect_latency_spike([], baseline_p99_ms=30.0)
        assert not report.is_anomalous
        assert report.sample_size == 0

    def test_threshold_calculation(self, detector: AnomalyDetector) -> None:
        report = detector.detect_latency_spike([50.0], baseline_p99_ms=30.0)
        assert report.threshold == 60.0


class TestErrorRate:
    def test_low_error_rate(self, detector: AnomalyDetector) -> None:
        report = detector.detect_error_rate(total_requests=1000, error_count=10)
        assert report.anomaly_type == AnomalyType.ERROR_RATE
        assert not report.is_anomalous
        assert report.score == pytest.approx(0.01)

    def test_high_error_rate(self, detector: AnomalyDetector) -> None:
        report = detector.detect_error_rate(total_requests=100, error_count=20)
        assert report.is_anomalous
        assert report.score == pytest.approx(0.20)

    def test_zero_requests(self, detector: AnomalyDetector) -> None:
        report = detector.detect_error_rate(total_requests=0, error_count=0)
        assert not report.is_anomalous
        assert report.sample_size == 0

    def test_boundary_threshold(self, detector: AnomalyDetector) -> None:
        report = detector.detect_error_rate(total_requests=100, error_count=5)
        assert not report.is_anomalous
        report2 = detector.detect_error_rate(total_requests=100, error_count=6)
        assert report2.is_anomalous
