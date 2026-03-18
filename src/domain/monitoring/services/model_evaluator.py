import numpy as np


class ModelEvaluator:
    """
    Domain Service responsible for evaluating model performance.
    Computes metrics like Accuracy, Precision, Recall, and F1-score.
    """

    def evaluate(self, y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        """
        Evaluate performance metrics given true labels and predictions.
        """
        if not y_true or not y_pred:
            raise ValueError("Evaluation data cannot be empty")

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        accuracy = np.mean(y_true_arr == y_pred_arr)

        # Calculate Precision, Recall, F1 for binary classification (class 1)
        tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
        fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
        fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    def is_better(
        self,
        champion_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
        primary_metric: str = "f1_score",
    ) -> bool:
        """
        Compare two models. Returns True if challenger is better.
        """
        challenger_val = challenger_metrics.get(primary_metric, 0.0)
        champion_val = champion_metrics.get(primary_metric, 0.0)
        return challenger_val > champion_val
