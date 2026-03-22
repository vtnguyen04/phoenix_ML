"""A/B test statistical significance analyzer.

Computes: chi-squared test, Mann-Whitney U, confidence intervals, effect size.
Used with existing canary/shadow routing strategies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _get_thresholds() -> tuple[float, float]:
    """Load thresholds from config."""
    from phoenix_ml.config import get_settings  # noqa: PLC0415

    s = get_settings()
    return s.AB_TEST_NEGLIGIBLE_EFFECT, s.AB_TEST_LARGE_EFFECT


@dataclass
class ABTestResult:
    """Result of A/B test statistical analysis."""

    test_name: str
    control_name: str
    variant_name: str

    control_mean: float
    variant_mean: float
    control_count: int
    variant_count: int

    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float
    recommendation: str

    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "control": {
                "name": self.control_name,
                "mean": round(self.control_mean, 4),
                "count": self.control_count,
            },
            "variant": {
                "name": self.variant_name,
                "mean": round(self.variant_mean, 4),
                "count": self.variant_count,
            },
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "effect_size": round(self.effect_size, 4),
            "recommendation": self.recommendation,
        }


class ABTestAnalyzer:
    """Statistical analysis for A/B testing model variants.

    Methods:
    - compare_means: compare numeric outcomes (latency, scores)
    - compare_proportions: compare success rates (accuracy)
    """

    def __init__(self, confidence_level: float | None = None) -> None:
        from phoenix_ml.config import get_settings  # noqa: PLC0415

        s = get_settings()
        self.confidence_level = (
            confidence_level
            if confidence_level is not None
            else s.AB_TEST_CONFIDENCE_LEVEL
        )
        self._alpha = 1.0 - self.confidence_level

    def compare_means(
        self,
        control: list[float],
        variant: list[float],
        test_name: str = "model_comparison",
        control_name: str = "champion",
        variant_name: str = "challenger",
    ) -> ABTestResult:
        """Compare means using Mann-Whitney U test (non-parametric)."""
        c = np.array(control, dtype=np.float64)
        v = np.array(variant, dtype=np.float64)

        c_mean = float(np.mean(c))
        v_mean = float(np.mean(v))

        # Mann-Whitney U statistic
        p_value = self._mann_whitney_u(c, v)

        # Cohen's d effect size
        pooled_std = math.sqrt(
            (float(np.std(c, ddof=1)) ** 2 + float(np.std(v, ddof=1)) ** 2) / 2
        )
        effect_size = abs(c_mean - v_mean) / pooled_std if pooled_std > 0 else 0

        is_sig = p_value < self._alpha

        recommendation = self._recommend(
            is_sig, v_mean > c_mean, effect_size
        )

        return ABTestResult(
            test_name=test_name,
            control_name=control_name,
            variant_name=variant_name,
            control_mean=c_mean,
            variant_mean=v_mean,
            control_count=len(c),
            variant_count=len(v),
            p_value=p_value,
            is_significant=is_sig,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            recommendation=recommendation,
            details={"test_method": "mann_whitney_u"},
        )

    def compare_proportions(
        self,
        control_successes: int,
        control_total: int,
        variant_successes: int,
        variant_total: int,
        test_name: str = "accuracy_comparison",
        control_name: str = "champion",
        variant_name: str = "challenger",
    ) -> ABTestResult:
        """Compare proportions using chi-squared test."""
        p1 = control_successes / control_total if control_total > 0 else 0
        p2 = variant_successes / variant_total if variant_total > 0 else 0

        # Chi-squared 2x2 contingency table
        p_value = self._chi_squared_proportions(
            control_successes,
            control_total,
            variant_successes,
            variant_total,
        )

        # Effect size (Cohen's h)
        effect_size = abs(
            2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
        )

        is_sig = p_value < self._alpha
        recommendation = self._recommend(is_sig, p2 > p1, effect_size)

        return ABTestResult(
            test_name=test_name,
            control_name=control_name,
            variant_name=variant_name,
            control_mean=p1,
            variant_mean=p2,
            control_count=control_total,
            variant_count=variant_total,
            p_value=p_value,
            is_significant=is_sig,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            recommendation=recommendation,
            details={"test_method": "chi_squared"},
        )

    def _mann_whitney_u(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Mann-Whitney U test p-value (two-sided)."""
        combined = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(combined)) + 1.0
        n1, n2 = len(x), len(y)
        r1 = float(ranks[:n1].sum())
        u1 = r1 - n1 * (n1 + 1) / 2
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if sigma == 0:
            return 1.0
        z = abs(u1 - mu) / sigma
        # Approximate p-value from z-score (two-sided)
        return 2 * (1 - self._norm_cdf(z))

    def _chi_squared_proportions(
        self, s1: int, n1: int, s2: int, n2: int
    ) -> float:
        """Compute chi-squared p-value for 2x2 contingency."""
        f1, f2 = n1 - s1, n2 - s2
        total = n1 + n2
        if total == 0:
            return 1.0

        # Expected frequencies
        exp_s1 = n1 * (s1 + s2) / total
        exp_f1 = n1 * (f1 + f2) / total
        exp_s2 = n2 * (s1 + s2) / total
        exp_f2 = n2 * (f1 + f2) / total

        chi2 = 0.0
        for obs, exp in [(s1, exp_s1), (f1, exp_f1), (s2, exp_s2), (f2, exp_f2)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

        # Approximate p-value from chi-squared (1 df)
        return 1 - self._chi2_cdf(chi2)

    @staticmethod
    def _norm_cdf(z: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    @staticmethod
    def _chi2_cdf(x: float, df: int = 1) -> float:
        """Chi-squared CDF approximation (1 degree of freedom)."""
        if x <= 0:
            return 0.0
        # For df=1: chi2 CDF = 2 * norm_cdf(sqrt(x)) - 1
        return math.erf(math.sqrt(x / 2))

    @staticmethod
    def _recommend(
        is_significant: bool, variant_better: bool, effect_size: float
    ) -> str:
        negligible, large = _get_thresholds()
        if not is_significant:
            return "No significant difference. Continue testing or keep control."
        if effect_size < negligible:
            return "Statistically significant but negligible effect. Keep control."
        if variant_better:
            if effect_size > large:
                return "Strong evidence: promote variant to production."
            return "Moderate evidence: consider promoting variant."
        if effect_size > large:
            return "Strong evidence: variant is worse. Keep control."
        return "Moderate evidence: variant may be worse. Keep control."
