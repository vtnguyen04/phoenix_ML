"""Tests for Optuna hyperparameter optimizer."""

import numpy as np
import pytest

from src.domain.training.services.optuna_optimizer import OptunaOptimizer


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    x = rng.randn(200, 5).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    x = rng.randn(200, 5).astype(np.float32)
    y = x[:, 0] * 2 + x[:, 1] + rng.randn(200).astype(np.float32) * 0.1
    return x, y


class TestOptunaOptimizer:
    def test_classification_optimize(
        self, classification_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = classification_data
        optimizer = OptunaOptimizer(task="classification", n_trials=5, cv_folds=3)
        result = optimizer.optimize(x, y)

        assert result.best_score > 0.5  # Should beat random
        assert "n_estimators" in result.best_params
        assert "max_depth" in result.best_params
        assert result.n_trials == 5

    def test_regression_optimize(
        self, regression_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = regression_data
        optimizer = OptunaOptimizer(task="regression", n_trials=5, cv_folds=3, metric="r2")
        result = optimizer.optimize(x, y)

        assert result.best_score > 0  # Should have positive R²
        assert result.optimization_metric == "r2"

    def test_all_trials_collected(
        self, classification_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = classification_data
        optimizer = OptunaOptimizer(task="classification", n_trials=3, cv_folds=2)
        result = optimizer.optimize(x, y)

        assert len(result.all_trials) == 3
        for trial in result.all_trials:
            assert "params" in trial
            assert "score" in trial

    def test_custom_search_space(
        self, classification_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = classification_data
        space = {
            "n_estimators": {"type": "int", "low": 50, "high": 100},
            "max_depth": {"type": "int", "low": 3, "high": 5},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        }
        optimizer = OptunaOptimizer(task="classification", n_trials=3, cv_folds=2)
        result = optimizer.optimize(x, y, search_space=space)

        assert 50 <= result.best_params["n_estimators"] <= 100
        assert 3 <= result.best_params["max_depth"] <= 5

    def test_to_dict(self, classification_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = classification_data
        optimizer = OptunaOptimizer(task="classification", n_trials=3, cv_folds=2)
        result = optimizer.optimize(x, y)

        d = result.to_dict()
        assert "best_params" in d
        assert "best_score" in d
        assert "top_5_trials" in d
