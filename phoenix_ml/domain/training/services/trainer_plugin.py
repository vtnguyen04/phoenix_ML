"""Abstract interface for model training.

Implementations handle training and validation for a specific ML
framework (scikit-learn, PyTorch, etc.) and are registered in
``PluginRegistry`` keyed by ``model_id``.

Methods:
    train(config) -> TrainResult: Execute training, return result with model path.
    validate(model_path, data_path) -> dict[str, float]: Evaluate a saved model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainResult:
    """Output of a training run.

    Attributes:
        model_path: Path to the exported model artifact.
        metrics: Evaluation metrics (accuracy, mAP, RMSE, etc.).
        metadata: Additional info (feature names, class labels, etc.).
    """

    model_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ITrainer(ABC):
    """Abstract base for model training.

    Subclasses implement ``train()`` and ``validate()`` for a specific
    ML framework. Registered in ``PluginRegistry`` per ``model_id``.
    """

    @abstractmethod
    async def train(self, config: dict[str, Any]) -> TrainResult:
        """Train a model and return the result.

        Args:
            config: Training configuration dict. Contents vary by problem type.
                Common keys: dataset_path, epochs, batch_size, learning_rate.
                Custom keys for specific problems (e.g., img_size for YOLO).

        Returns:
            TrainResult with model artifact path and metrics.
        """
        ...

    @abstractmethod
    async def validate(self, model_path: str, data_path: str) -> dict[str, float]:
        """Validate a trained model on a dataset.

        Args:
            model_path: Path to the model artifact.
            data_path: Path to validation data.

        Returns:
            Dict of metric_name -> value (e.g. {"accuracy": 0.95}).
        """
        ...

    async def export(self, model_path: str, export_format: str = "onnx") -> str:
        """Export model to a deployment-ready format.

        Default implementation returns model_path unchanged.
        Override for frameworks that need explicit export (e.g. PyTorch → ONNX).

        Args:
            model_path: Path to the trained model.
            export_format: Target format ("onnx", "tensorrt", "torchscript").

        Returns:
            Path to the exported model artifact.
        """
        return model_path
