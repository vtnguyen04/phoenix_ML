"""
Drift Calculator — Strategy Pattern.

Each drift detection algorithm is a DriftStrategy.
Adding a new test = 1 new class + 1 dict entry.
All thresholds read from config.py (zero magic numbers).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from phoenix_ml.config import get_settings
from phoenix_ml.domain.monitoring.entities.drift_report import DriftReport

_settings = get_settings()


@dataclass(frozen=True)
class DriftConfig:
    """All drift-related thresholds — read from centralized config."""

    psi_no_drift: float = _settings.DRIFT_PSI_NO_DRIFT
    psi_moderate: float = _settings.DRIFT_PSI_MODERATE
    wasserstein_factor: float = _settings.DRIFT_WASSERSTEIN_FACTOR
    critical_threshold: float = _settings.DRIFT_CRITICAL_THRESHOLD
    chi2_bins: int = _settings.DRIFT_CHI2_BINS


# ── Strategy Interface ────────────────────────────────────────────


class DriftStrategy(ABC):
    """Abstract strategy for drift detection algorithms."""

    @abstractmethod
    def calculate(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float,
    ) -> DriftReport:
        """Calculate drift between reference and current distributions."""


# ── Concrete Strategies ──────────────────────────────────────────


_cfg = DriftConfig()


def _generate_recommendation(
    drift_detected: bool,
    score: float,
    feature_name: str,
) -> str:
    if not drift_detected:
        return "No action needed. Distribution remains stable."
    if score > _cfg.critical_threshold:
        return (
            f"CRITICAL: Severe drift in {feature_name}. "
            "Immediate retraining and pipeline check required."
        )
    return f"WARNING: Drift detected in {feature_name}. Scheduling auto-retraining pipeline."


class KSDriftStrategy(DriftStrategy):
    """Kolmogorov-Smirnov two-sample test."""

    def calculate(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.05,
    ) -> DriftReport:
        result: Any = stats.ks_2samp(reference, current)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        drift_detected = bool(p_value < threshold)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=statistic,
            threshold=threshold,
            method="ks_test",
            recommendation=_generate_recommendation(drift_detected, statistic, feature_name),
            sample_size=len(current),
        )


class PSIDriftStrategy(DriftStrategy):
    """Population Stability Index."""

    def __init__(self, n_bins: int = _cfg.chi2_bins) -> None:
        self._n_bins = n_bins

    def calculate(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = _cfg.psi_moderate,
    ) -> DriftReport:
        num_bins = self._n_bins
        bins = np.percentile(reference, np.linspace(0, 100, num_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:  # noqa: PLR2004
            bins = np.array([-np.inf, np.inf])
        else:
            bins[0] = -np.inf
            bins[-1] = np.inf

        ref_counts = np.histogram(reference, bins=bins)[0] / len(reference)
        cur_counts = np.histogram(current, bins=bins)[0] / len(current)

        eps = 1e-10
        ref_counts = np.clip(ref_counts, eps, 1.0)
        cur_counts = np.clip(cur_counts, eps, 1.0)

        psi = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
        drift_detected = bool(psi > _cfg.psi_moderate)
        p_value = 0.01 if drift_detected else (0.1 if psi > _cfg.psi_no_drift else 0.5)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=psi,
            threshold=_cfg.psi_moderate,
            method="psi",
            recommendation=_generate_recommendation(drift_detected, psi, feature_name),
            sample_size=len(current),
        )


class WassersteinDriftStrategy(DriftStrategy):
    """Earth Mover's Distance with bootstrap permutation p-value."""

    def __init__(self, n_bootstrap: int = 1000) -> None:
        self._n_bootstrap = n_bootstrap

    def calculate(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.05,
    ) -> DriftReport:
        distance = float(stats.wasserstein_distance(reference, current))
        adaptive_threshold = float(np.std(reference) * _cfg.wasserstein_factor)

        combined = np.concatenate([reference, current])
        n_ref = len(reference)
        rng = np.random.default_rng(seed=42)
        bootstrap_distances = np.empty(self._n_bootstrap)
        for i in range(self._n_bootstrap):
            perm = rng.permutation(combined)
            bootstrap_distances[i] = stats.wasserstein_distance(perm[:n_ref], perm[n_ref:])

        p_value = float(np.mean(bootstrap_distances >= distance))
        drift_detected = bool(distance > adaptive_threshold)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=distance,
            threshold=adaptive_threshold,
            method="wasserstein",
            recommendation=_generate_recommendation(drift_detected, distance, feature_name),
            sample_size=len(current),
        )


class Chi2DriftStrategy(DriftStrategy):
    """Chi-squared test for categorical/binned continuous features."""

    def calculate(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.05,
    ) -> DriftReport:
        combined = np.concatenate([reference, current])
        bins = np.histogram_bin_edges(combined, bins=_cfg.chi2_bins)

        ref_counts = np.histogram(reference, bins=bins)[0].astype(float) + 1.0
        cur_counts = np.histogram(current, bins=bins)[0].astype(float) + 1.0

        ref_proportions = ref_counts / ref_counts.sum()
        expected = ref_proportions * cur_counts.sum()

        result: Any = stats.chisquare(cur_counts, f_exp=expected)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        drift_detected = bool(p_value < threshold)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=statistic,
            threshold=threshold,
            method="chi2",
            recommendation=_generate_recommendation(drift_detected, statistic, feature_name),
            sample_size=len(current),
        )


# ── Strategy Registry (OCP: add new tests via dict entry) ────────

_DRIFT_STRATEGIES: dict[str, DriftStrategy] = {
    "ks": KSDriftStrategy(),
    "psi": PSIDriftStrategy(),
    "wasserstein": WassersteinDriftStrategy(),
    "chi2": Chi2DriftStrategy(),
}


# ── DriftCalculator (Context) ────────────────────────────────────


class DriftCalculator:
    """
    Domain Service — delegates to the appropriate DriftStrategy.
    Zero if/elif chains. Add new test = 1 class + 1 dict entry.
    """

    def calculate_drift(
        self,
        feature_name: str,
        reference_data: list[float],
        current_data: list[float],
        threshold: float = 0.05,
        test_type: str = "ks",
    ) -> DriftReport:
        if not reference_data or not current_data:
            raise ValueError("Data samples cannot be empty")

        strategy = _DRIFT_STRATEGIES.get(test_type)
        if strategy is None:
            supported = ", ".join(sorted(_DRIFT_STRATEGIES))
            raise ValueError(f"Unsupported test type: '{test_type}'. Supported: {supported}")

        return strategy.calculate(
            feature_name=feature_name,
            reference=np.array(reference_data),
            current=np.array(current_data),
            threshold=threshold,
        )
