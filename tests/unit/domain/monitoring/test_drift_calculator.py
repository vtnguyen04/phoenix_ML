import numpy as np
import pytest

from src.domain.monitoring.services.drift_calculator import DriftCalculator


@pytest.fixture
def calculator() -> DriftCalculator:
    return DriftCalculator()


def test_ks_test_no_drift(calculator: DriftCalculator) -> None:
    # Two identical distributions
    reference = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    current = [1.0, 2.0, 3.0, 4.0, 5.0] * 10

    report = calculator.calculate_drift(
        feature_name="feat1",
        reference_data=reference,
        current_data=current,
        test_type="ks",
        threshold=0.05,
    )

    assert report.drift_detected is False
    assert report.p_value > 0.05
    assert "No action needed" in report.recommendation


def test_ks_test_with_drift(calculator: DriftCalculator) -> None:
    # Significant shift
    reference = np.random.normal(0, 1, 100).tolist()
    current = np.random.normal(5, 1, 100).tolist()

    report = calculator.calculate_drift(
        feature_name="feat1", reference_data=reference, current_data=current, test_type="ks"
    )

    assert report.drift_detected is True
    assert report.p_value < 0.05
    assert "drift" in report.recommendation.lower()


def test_psi_test_no_drift(calculator: DriftCalculator) -> None:
    reference = np.random.normal(10, 2, 1000).tolist()
    current = np.random.normal(10, 2, 1000).tolist()

    report = calculator.calculate_drift(
        feature_name="feat1", reference_data=reference, current_data=current, test_type="psi"
    )

    assert report.drift_detected is False
    assert report.statistic < 0.1


def test_psi_test_with_drift(calculator: DriftCalculator) -> None:
    reference = np.random.normal(10, 2, 1000).tolist()
    current = np.random.normal(15, 2, 1000).tolist()

    report = calculator.calculate_drift(
        feature_name="feat1", reference_data=reference, current_data=current, test_type="psi"
    )

    assert report.drift_detected is True
    assert report.statistic > 0.25


def test_wasserstein_test(calculator: DriftCalculator) -> None:
    reference = [1.0, 1.0, 1.0]
    current = [10.0, 10.0, 10.0]

    report = calculator.calculate_drift(
        feature_name="feat1",
        reference_data=reference,
        current_data=current,
        test_type="wasserstein",
    )

    assert report.drift_detected is True
    assert report.statistic == 9.0


def test_invalid_test_type(calculator: DriftCalculator) -> None:
    with pytest.raises(ValueError, match="Unsupported test type"):
        calculator.calculate_drift("f", [1], [1], test_type="invalid")


def test_empty_data(calculator: DriftCalculator) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        calculator.calculate_drift("f", [], [1])
