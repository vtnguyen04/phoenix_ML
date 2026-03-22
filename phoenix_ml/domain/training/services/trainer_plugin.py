"""
ITrainer — Plugin Interface for Model Training.

Enables the Phoenix ML framework to support any ML training pipeline
(scikit-learn, PyTorch, TensorFlow, YOLOv8, HuggingFace, etc.)
without modifying the framework core.

Users implement this interface for their specific ML problem:
  - Classification: GradientBoosting, XGBoost, LightGBM
  - Object Detection: YOLOv8, Faster R-CNN, DETR
  - Recommendation: Matrix Factorization, Neural CF
  - Time Series: Prophet, LSTM, Transformer
  - NLP: BERT, GPT fine-tuning
  - Custom: Any other ML/DL framework

The self-healing pipeline calls ITrainer.train() to retrain
when drift is detected, making retraining fully pluggable.
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
    """Plugin interface for model training.

    Implement this for your specific ML problem type.
    The self-healing pipeline uses this to retrain models
    when drift is detected.

    Example::

        class YOLOTrainer(ITrainer):
            async def train(self, config):
                model = YOLO("yolov8n.pt")
                results = model.train(data=config["dataset_path"], epochs=100)
                model.export(format="onnx")
                return TrainResult(
                    model_path="runs/detect/train/weights/best.onnx",
                    metrics={"mAP50": results.maps[0]},
                )
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
