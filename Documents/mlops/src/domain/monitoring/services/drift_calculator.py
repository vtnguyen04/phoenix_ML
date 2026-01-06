from typing import Any

import numpy as np
from scipy import stats  # type: ignore

from src.domain.monitoring.entities.drift_report import DriftReport


class DriftCalculator:
    """
    Domain Service responsible for calculating statistical drift.
    Uses Kolmogorov-Smirnov (KS) test for continuous variables.
    """

    def calculate_drift(
        self,
        feature_name: str,
        reference_data: list[float],
        current_data: list[float],
        threshold: float = 0.05,
    ) -> DriftReport:
        """
        Performs KS Test.
        Null Hypothesis (H0): Two samples are drawn from the same distribution.
        If p_value < threshold, reject H0 -> Drift Detected.
        """
        if not reference_data or not current_data:
            raise ValueError("Data samples cannot be empty")

        # Convert to numpy for efficiency
        ref_arr = np.array(reference_data)
        cur_arr = np.array(current_data)

        # KS Test
        result: Any = stats.ks_2samp(ref_arr, cur_arr)
        statistic: float = float(result.statistic)
        p_value: float = float(result.pvalue)

        drift_detected = p_value < threshold

        return DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            p_value=p_value,
            statistic=statistic,
            threshold=threshold,
        )
