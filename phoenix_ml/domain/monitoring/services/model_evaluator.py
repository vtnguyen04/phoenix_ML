"""Abstract interface for task-specific model evaluation.

Implementations compute metrics appropriate for the task type
(classification → accuracy/F1, regression → RMSE/MAE, detection → mAP).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# ─── Interface ────────────────────────────────────────────────────


class IModelEvaluator(ABC):
    """Abstract base for model performance evaluation.

    Subclasses implement ``evaluate()`` and ``primary_metric()`` for
    a specific task type. ``is_better()`` defaults to higher-is-better.
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
    """Evaluator for binary/multiclass classification.

    Computes accuracy, precision, recall, F1-score. Primary metric: ``f1_score``.
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

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
    """Evaluator for regression tasks.

    Computes RMSE, MAE, R², MAPE. Primary metric: ``rmse`` (lower-is-better).
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
        mape = float(np.mean(np.abs(errors[mask] / y_t[mask])) * 100) if np.any(mask) else 0.0

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
    """Return an evaluator for the given task type.

    Args:
        task_type: One of ``classification``, ``binary_classification``,
            ``regression``, ``timeseries``. Unknown types fall back to
            ``ClassificationEvaluator``.

    Returns:
        An ``IModelEvaluator`` instance.
    """
    evaluator_cls = _EVALUATOR_MAP.get(task_type, ClassificationEvaluator)
    return evaluator_cls()
