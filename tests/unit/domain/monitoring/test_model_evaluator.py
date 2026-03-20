"""Tests for IModelEvaluator and task-specific evaluator implementations."""

import pytest

from src.domain.monitoring.services.model_evaluator import (
    ClassificationEvaluator,
    IModelEvaluator,
    RegressionEvaluator,
    get_evaluator,
)

# ─── ClassificationEvaluator ─────────────────────────────────────


class TestClassificationEvaluator:
    def setup_method(self) -> None:
        self.evaluator = ClassificationEvaluator()

    def test_perfect_predictions(self) -> None:
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong(self) -> None:
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["recall"] == 0.0

    def test_mixed_results(self) -> None:
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["f1_score"] < 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            self.evaluator.evaluate([1, 0], [1])

    def test_primary_metric(self) -> None:
        assert self.evaluator.primary_metric() == "f1_score"

    def test_is_better_higher_f1(self) -> None:
        champion = {"f1_score": 0.80}
        challenger = {"f1_score": 0.85}
        assert self.evaluator.is_better(champion, challenger)

    def test_is_better_lower_f1(self) -> None:
        champion = {"f1_score": 0.90}
        challenger = {"f1_score": 0.85}
        assert not self.evaluator.is_better(champion, challenger)


# ─── RegressionEvaluator ─────────────────────────────────────────


class TestRegressionEvaluator:
    def setup_method(self) -> None:
        self.evaluator = RegressionEvaluator()

    def test_perfect_predictions(self) -> None:
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_imperfect_predictions(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.2, 2.8]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert 0 < metrics["r2"] < 1

    def test_mape_calculation(self) -> None:
        y_true = [100.0, 200.0]
        y_pred = [110.0, 180.0]
        metrics = self.evaluator.evaluate(y_true, y_pred)
        assert metrics["mape"] == pytest.approx(10.0, abs=0.1)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.evaluator.evaluate([], [])

    def test_primary_metric(self) -> None:
        assert self.evaluator.primary_metric() == "rmse"

    def test_is_better_lower_rmse(self) -> None:
        """Lower RMSE = better for regression."""
        champion = {"rmse": 0.50}
        challenger = {"rmse": 0.30}
        assert self.evaluator.is_better(champion, challenger)

    def test_is_better_higher_rmse(self) -> None:
        champion = {"rmse": 0.20}
        challenger = {"rmse": 0.30}
        assert not self.evaluator.is_better(champion, challenger)

    def test_is_better_r2_higher_wins(self) -> None:
        champion = {"r2": 0.80}
        challenger = {"r2": 0.90}
        assert self.evaluator.is_better(champion, challenger, primary_metric="r2")


# ─── Factory ─────────────────────────────────────────────────────


class TestGetEvaluator:
    def test_classification(self) -> None:
        evaluator = get_evaluator("classification")
        assert isinstance(evaluator, ClassificationEvaluator)

    def test_regression(self) -> None:
        evaluator = get_evaluator("regression")
        assert isinstance(evaluator, RegressionEvaluator)

    def test_timeseries(self) -> None:
        evaluator = get_evaluator("timeseries")
        assert isinstance(evaluator, RegressionEvaluator)

    def test_unknown_defaults_to_classification(self) -> None:
        evaluator = get_evaluator("object_detection")
        assert isinstance(evaluator, ClassificationEvaluator)

    def test_returns_imodel_evaluator(self) -> None:
        evaluator = get_evaluator("classification")
        assert isinstance(evaluator, IModelEvaluator)
