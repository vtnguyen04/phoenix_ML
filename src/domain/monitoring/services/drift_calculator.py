from typing import Any

import numpy as np
from scipy import stats

from src.domain.monitoring.entities.drift_report import DriftReport

PSI_NO_DRIFT = 0.1
PSI_MODERATE_DRIFT = 0.25
WASSERSTEIN_THRESHOLD_FACTOR = 0.5
CRITICAL_DRIFT_THRESHOLD = 0.5
CHI2_DEFAULT_BINS = 10


class DriftCalculator:
    """
    Domain Service responsible for calculating statistical drift.
    Supports KS-test, PSI, Wasserstein distance, and Chi-squared test.
    """

    def calculate_drift(
        self,
        feature_name: str,
        reference_data: list[float],
        current_data: list[float],
        threshold: float = 0.05,
        test_type: str = "ks",
    ) -> DriftReport:
        """
        Calculates drift between reference and current data using specified test.
        """
        if not reference_data or not current_data:
            raise ValueError("Data samples cannot be empty")

        ref_arr = np.array(reference_data)
        cur_arr = np.array(current_data)

        if test_type == "ks":
            return self._ks_test(feature_name, ref_arr, cur_arr, threshold)
        if test_type == "psi":
            return self._psi_test(feature_name, ref_arr, cur_arr)
        if test_type == "wasserstein":
            return self._wasserstein_test(feature_name, ref_arr, cur_arr)
        if test_type == "chi2":
            return self._chi2_test(feature_name, ref_arr, cur_arr, threshold)

        raise ValueError(f"Unsupported test type: {test_type}")

    def _ks_test(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float,
    ) -> DriftReport:
        result: Any = stats.ks_2samp(reference, current)
        statistic: float = float(result.statistic)
        p_value: float = float(result.pvalue)
        drift_detected = bool(p_value < threshold)

        recommendation = self._generate_recommendation(drift_detected, statistic, feature_name)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=statistic,
            threshold=threshold,
            method="ks_test",
            recommendation=recommendation,
            sample_size=len(current),
        )

    def _psi_test(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> DriftReport:
        """Population Stability Index"""
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
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

        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))

        drift_detected = bool(psi > PSI_MODERATE_DRIFT)
        p_value = 0.01 if drift_detected else (0.1 if psi > PSI_NO_DRIFT else 0.5)

        recommendation = self._generate_recommendation(drift_detected, psi, feature_name)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=psi,
            threshold=PSI_MODERATE_DRIFT,
            method="psi",
            recommendation=recommendation,
            sample_size=len(current),
        )

    def _wasserstein_test(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> DriftReport:
        """Earth Mover's Distance with bootstrap p-value."""
        distance = float(stats.wasserstein_distance(reference, current))

        threshold = float(np.std(reference) * WASSERSTEIN_THRESHOLD_FACTOR)

        # Bootstrap permutation test for p-value
        combined = np.concatenate([reference, current])
        n_ref = len(reference)
        rng = np.random.default_rng(seed=42)
        bootstrap_distances = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            perm = rng.permutation(combined)
            boot_ref = perm[:n_ref]
            boot_cur = perm[n_ref:]
            bootstrap_distances[i] = stats.wasserstein_distance(boot_ref, boot_cur)

        p_value = float(np.mean(bootstrap_distances >= distance))
        drift_detected = bool(distance > threshold)

        recommendation = self._generate_recommendation(drift_detected, distance, feature_name)

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=distance,
            threshold=threshold,
            method="wasserstein",
            recommendation=recommendation,
            sample_size=len(current),
        )

    def _chi2_test(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.05,
    ) -> DriftReport:
        """Chi-squared test for categorical/binned continuous features."""
        combined = np.concatenate([reference, current])
        bins = np.histogram_bin_edges(combined, bins=CHI2_DEFAULT_BINS)

        ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
        cur_counts = np.histogram(current, bins=bins)[0].astype(float)

        ref_counts += 1.0
        cur_counts += 1.0

        ref_proportions = ref_counts / ref_counts.sum()
        expected = ref_proportions * cur_counts.sum()

        result: Any = stats.chisquare(cur_counts, f_exp=expected)
        statistic: float = float(result.statistic)
        p_value: float = float(result.pvalue)
        drift_detected = bool(p_value < threshold)

        recommendation = self._generate_recommendation(
            drift_detected, statistic, feature_name
        )

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=statistic,
            threshold=threshold,
            method="chi2",
            recommendation=recommendation,
            sample_size=len(current),
        )

    def _generate_recommendation(
        self, drift_detected: bool, score: float, feature_name: str
    ) -> str:
        if not drift_detected:
            return "No action needed. Distribution remains stable."

        if score > CRITICAL_DRIFT_THRESHOLD:
            return (
                f"CRITICAL: Severe drift in {feature_name}. "
                "Immediate retraining and pipeline check required."
            )
        return f"WARNING: Drift detected in {feature_name}. Scheduling auto-retraining pipeline."

