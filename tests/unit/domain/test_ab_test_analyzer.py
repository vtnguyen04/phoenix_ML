"""Tests for A/B test statistical analyzer."""

import pytest

from phoenix_ml.domain.monitoring.services.ab_test_analyzer import ABTestAnalyzer


@pytest.fixture
def analyzer() -> ABTestAnalyzer:
    return ABTestAnalyzer(confidence_level=0.95)


class TestABTestAnalyzer:
    def test_compare_means_no_difference(
        self, analyzer: ABTestAnalyzer
    ) -> None:
        import numpy as np

        rng = np.random.RandomState(42)
        control = rng.normal(0.5, 0.1, 50).tolist()
        variant = rng.normal(0.5, 0.1, 50).tolist()
        result = analyzer.compare_means(control, variant)
        # With same distribution, should NOT be significant
        assert result.p_value > 0.01

    def test_compare_means_significant(
        self, analyzer: ABTestAnalyzer
    ) -> None:
        control = [0.5, 0.6, 0.5, 0.4, 0.5] * 30
        variant = [0.9, 0.95, 0.85, 0.92, 0.88] * 30
        result = analyzer.compare_means(control, variant)
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.variant_mean > result.control_mean

    def test_compare_proportions_no_difference(
        self, analyzer: ABTestAnalyzer
    ) -> None:
        result = analyzer.compare_proportions(50, 100, 48, 100)
        assert not result.is_significant
        assert result.p_value > 0.05

    def test_compare_proportions_significant(
        self, analyzer: ABTestAnalyzer
    ) -> None:
        result = analyzer.compare_proportions(90, 100, 60, 100)
        assert result.is_significant
        assert result.p_value < 0.05

    def test_to_dict_format(self, analyzer: ABTestAnalyzer) -> None:
        result = analyzer.compare_means(
            [1.0, 2.0, 3.0] * 10,
            [4.0, 5.0, 6.0] * 10,
            test_name="latency_test",
            control_name="model_a",
            variant_name="model_b",
        )
        d = result.to_dict()
        assert d["test_name"] == "latency_test"
        assert d["control"]["name"] == "model_a"
        assert d["variant"]["name"] == "model_b"
        assert "p_value" in d
        assert "recommendation" in d

    def test_recommendation_promote(self, analyzer: ABTestAnalyzer) -> None:
        control = [0.5] * 100
        variant = [0.9] * 100
        result = analyzer.compare_means(control, variant)
        assert "promote" in result.recommendation.lower()

    def test_recommendation_keep_control(
        self, analyzer: ABTestAnalyzer
    ) -> None:
        control = [0.9] * 100
        variant = [0.5] * 100
        result = analyzer.compare_means(control, variant)
        assert "keep control" in result.recommendation.lower()

    def test_effect_size_calculated(
        self, analyzer: ABTestAnalyzer
    ) -> None:
        result = analyzer.compare_means(
            [1.0, 2.0] * 50, [5.0, 6.0] * 50
        )
        assert result.effect_size > 0
