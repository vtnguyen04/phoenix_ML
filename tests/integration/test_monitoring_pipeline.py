"""
Integration Test: Monitoring Pipeline.

Tests the monitoring flow independently from the self-healing flow:
drift detection, performance reporting, and report history.
"""

import numpy as np
import pytest

from phoenix_ml.domain.monitoring.services.drift_calculator import DriftCalculator


@pytest.fixture
def drift_calculator() -> DriftCalculator:
    return DriftCalculator()


class TestMonitoringPipeline:
    """Integration tests for the monitoring pipeline."""

    def test_ks_test_no_drift_same_distribution(self, drift_calculator: DriftCalculator) -> None:
        """KS test on same distribution should report no drift."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1, size=500).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="income",
            reference_data=data[:250],
            current_data=data[250:],
            test_type="ks",
        )

        assert report.drift_detected is False
        assert report.p_value > 0.05
        assert report.method == "ks_test"
        assert "No action" in report.recommendation

    def test_ks_test_detects_shifted_distribution(self, drift_calculator: DriftCalculator) -> None:
        """KS test detects drift when distribution is shifted."""
        rng = np.random.default_rng(42)
        reference = rng.normal(loc=0, scale=1, size=500).tolist()
        shifted = rng.normal(loc=3, scale=1, size=500).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="income",
            reference_data=reference,
            current_data=shifted,
            test_type="ks",
        )

        assert report.drift_detected is True
        assert report.p_value < 0.05

    def test_psi_test_no_drift(self, drift_calculator: DriftCalculator) -> None:
        """PSI on similar distributions should report no drift."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1, size=1000).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="age",
            reference_data=data[:500],
            current_data=data[500:],
            test_type="psi",
        )

        assert report.drift_detected is False
        assert report.method == "psi"

    def test_psi_test_detects_drift(self, drift_calculator: DriftCalculator) -> None:
        """PSI detects drift with heavily shifted distribution."""
        rng = np.random.default_rng(42)
        reference = rng.normal(loc=0, scale=1, size=500).tolist()
        shifted = rng.normal(loc=5, scale=2, size=500).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="age",
            reference_data=reference,
            current_data=shifted,
            test_type="psi",
        )

        assert report.drift_detected is True

    def test_wasserstein_detects_drift(self, drift_calculator: DriftCalculator) -> None:
        """Wasserstein distance flags significant shift."""
        rng = np.random.default_rng(42)
        reference = rng.normal(loc=0, scale=1, size=500).tolist()
        shifted = rng.normal(loc=10, scale=1, size=500).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="credit_score",
            reference_data=reference,
            current_data=shifted,
            test_type="wasserstein",
        )

        assert report.drift_detected is True
        assert report.method == "wasserstein"

    def test_chi2_test_no_drift(self, drift_calculator: DriftCalculator) -> None:
        """Chi-squared test on same distribution reports no drift."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1, size=1000).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="duration",
            reference_data=data[:500],
            current_data=data[500:],
            test_type="chi2",
        )

        assert report.drift_detected is False
        assert report.method == "chi2"

    def test_empty_data_raises(self, drift_calculator: DriftCalculator) -> None:
        """Empty data samples should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            drift_calculator.calculate_drift(
                feature_name="x",
                reference_data=[],
                current_data=[1.0, 2.0],
                test_type="ks",
            )

    def test_recommendation_severity_levels(self, drift_calculator: DriftCalculator) -> None:
        """Verify recommendation text reflects drift severity."""
        rng = np.random.default_rng(42)
        reference = rng.normal(loc=0, scale=1, size=500).tolist()
        severe_shift = rng.normal(loc=10, scale=1, size=500).tolist()

        report = drift_calculator.calculate_drift(
            feature_name="amount",
            reference_data=reference,
            current_data=severe_shift,
            test_type="ks",
        )

        assert report.drift_detected is True
        assert "CRITICAL" in report.recommendation or "WARNING" in report.recommendation
