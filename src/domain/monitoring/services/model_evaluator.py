"""
Model Evaluator Plugin — Task-Agnostic Evaluation Interface.

Provides IModelEvaluator interface and task-specific implementations.
Each ML problem type computes its own metrics.

Examples:
    - Classification: accuracy, precision, recall, F1
    - Regression: RMSE, MAE, R²
    - Detection: mAP, mAP50
    - Recommendation: NDCG, MRR
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# ─── Interface ────────────────────────────────────────────────────


class IModelEvaluator(ABC):
    """
    Interface for model performance evaluation.

    Each ML problem type should implement this to compute
    task-appropriate metrics and comparison logic.
    """

    @abstractmethod
    def evaluate(
        self,
        y_true: list[Any],
        y_pred: list[Any],
    ) -> dict[str, float]:
        """
        Compute task-specific performance metrics.

        Args:
            y_true: Ground truth values.
            y_pred: Model predictions.

        Returns:
            Dictionary of metric names to values.
        """

    @abstractmethod
    def primary_metric(self) -> str:
        """
        Return the primary metric name for this task type.
        Used for champion/challenger comparison.
        """

    def is_better(
        self,
        champion_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
        primary_metric: str | None = None,
    ) -> bool:
        """
        Compare two models. Returns True if challenger is better.

        Default: higher is better. Override for metrics where
        lower is better (e.g., RMSE).
        """
        metric = primary_metric or self.primary_metric()
        challenger_val = challenger_metrics.get(metric, 0.0)
        champion_val = champion_metrics.get(metric, 0.0)
        return challenger_val > champion_val


# ─── Classification ──────────────────────────────────────────────


class ClassificationEvaluator(IModelEvaluator):
    """
    Evaluator for classification tasks (binary and multiclass).
    Metrics: accuracy, precision, recall, F1-score.
    """

    def evaluate(
        self,
        y_true: list[Any],
        y_pred: list[Any],
    ) -> dict[str, float]:
        if not y_true or not y_pred:
            raise ValueError("Evaluation data cannot be empty")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        accuracy = float(np.mean(y_true_arr == y_pred_arr))

        # Binary classification metrics (class=1 as positive)
        tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    def primary_metric(self) -> str:
        return "f1_score"


# ─── Regression ──────────────────────────────────────────────────


class RegressionEvaluator(IModelEvaluator):
    """
    Evaluator for regression tasks.
    Metrics: RMSE, MAE, R², MAPE.
    """

    def evaluate(
        self,
        y_true: list[Any],
        y_pred: list[Any],
    ) -> dict[str, float]:
        if not y_true or not y_pred:
            raise ValueError("Evaluation data cannot be empty")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)

        errors = y_t - y_p
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))

        # R² (coefficient of determination)
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # MAPE (avoid division by zero)
        mask = y_t != 0
        mape = (
            float(np.mean(np.abs(errors[mask] / y_t[mask])) * 100)
            if np.any(mask)
            else 0.0
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }

    def primary_metric(self) -> str:
        return "rmse"

    def is_better(
        self,
        champion_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
        primary_metric: str | None = None,
    ) -> bool:
        """
        For regression, lower RMSE/MAE is better.
        For R², higher is better.
        """
        metric = primary_metric or self.primary_metric()
        challenger_val = challenger_metrics.get(metric, float("inf"))
        champion_val = champion_metrics.get(metric, float("inf"))

        # Lower-is-better metrics
        if metric in ("rmse", "mae", "mape"):
            return challenger_val < champion_val
        # Higher-is-better (r2)
        return challenger_val > champion_val


# ─── Factory ─────────────────────────────────────────────────────

_EVALUATOR_MAP: dict[str, type[IModelEvaluator]] = {
    "classification": ClassificationEvaluator,
    "binary_classification": ClassificationEvaluator,
    "regression": RegressionEvaluator,
    "timeseries": RegressionEvaluator,
}


def get_evaluator(task_type: str) -> IModelEvaluator:
    """
    Factory: returns the appropriate evaluator for a task type.

    Falls back to ClassificationEvaluator for unknown types.
    """
    evaluator_cls = _EVALUATOR_MAP.get(
        task_type, ClassificationEvaluator
    )
    return evaluator_cls()
