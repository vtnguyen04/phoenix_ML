"""
Tests for ModelEvaluator — domain service for computing model performance.
"""

import pytest

from src.domain.monitoring.services.model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """Unit tests for ModelEvaluator.evaluate and ModelEvaluator.is_better."""

    def setup_method(self) -> None:
        self.evaluator = ModelEvaluator()

    # ── evaluate ─────────────────────────────────────────

    def test_evaluate_perfect_predictions(self) -> None:
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_evaluate_all_wrong(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_evaluate_mixed(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["f1_score"] < 1
        assert metrics["recall"] == pytest.approx(2 / 3, rel=1e-3)

    def test_evaluate_all_positive_true(self) -> None:
        y_true = [1, 1, 1]
        y_pred = [1, 1, 0]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == pytest.approx(2 / 3, rel=1e-3)

    def test_evaluate_no_positive_predictions(self) -> None:
        y_true = [1, 1, 0]
        y_pred = [0, 0, 0]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_evaluate_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate([], [])

    def test_evaluate_mismatched_length_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            self.evaluator.evaluate([1, 0], [1])

    # ── is_better ────────────────────────────────────────

    def test_is_better_challenger_wins(self) -> None:
        champion = {"f1_score": 0.7, "accuracy": 0.8}
        challenger = {"f1_score": 0.85, "accuracy": 0.75}
        assert self.evaluator.is_better(champion, challenger) is True

    def test_is_better_champion_wins(self) -> None:
        champion = {"f1_score": 0.9, "accuracy": 0.85}
        challenger = {"f1_score": 0.7, "accuracy": 0.90}
        assert self.evaluator.is_better(champion, challenger) is False

    def test_is_better_equal_returns_false(self) -> None:
        metrics = {"f1_score": 0.8}
        assert self.evaluator.is_better(metrics, metrics) is False

    def test_is_better_custom_metric(self) -> None:
        champion = {"accuracy": 0.7}
        challenger = {"accuracy": 0.9}
        assert self.evaluator.is_better(champion, challenger, "accuracy") is True

    def test_is_better_missing_metric_uses_zero(self) -> None:
        champion: dict[str, float] = {}
        challenger = {"f1_score": 0.5}
        assert self.evaluator.is_better(champion, challenger) is True
