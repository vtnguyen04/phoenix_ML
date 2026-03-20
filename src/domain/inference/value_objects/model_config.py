"""
ModelConfig — Value Object for pluggable model definitions.

Enables the Phoenix ML framework to support any ML problem type
(classification, object detection, recommendation, time series, etc.)
without hardcoding dataset or feature information.

Users define their model via YAML config files in MODEL_CONFIG_DIR.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration describing a model and its problem domain.

    This value object decouples the framework from any specific ML task.
    All model-specific information (features, dataset, training script)
    is loaded from external config files, making the platform truly
    model-agnostic.

    Attributes:
        model_id: Unique identifier for the model (e.g. "fraud-detect").
        version: Model version string (e.g. "v1", "v1710234567").
        framework: Inference engine type ("onnx", "tensorrt", "triton", "pytorch").
        model_path: Relative path to model artifact from project root.
        task_type: ML problem type for documentation and routing.
            Supported values: "classification", "regression",
            "object_detection", "recommendation", "timeseries",
            "nlp", "embedding", "custom".
        feature_names: Ordered list of input feature names.
            Can be empty for models that don't use named features
            (e.g. image-based object detection).
        metadata: Additional model-specific metadata (metrics, dataset, etc.).
        dataset_name: Human-readable dataset identifier for lineage.
        train_script: Path to training script for self-healing retrain.
    """

    model_id: str
    version: str = "v1"
    framework: str = "onnx"
    model_path: str = ""
    task_type: str = "classification"
    feature_names: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    dataset_name: str = ""
    train_script: str = ""

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata as a mutable dict for serialization."""
        return dict(self.metadata)

    def with_version(self, version: str) -> "ModelConfig":
        """Return a new config with updated version."""
        return ModelConfig(
            model_id=self.model_id,
            version=version,
            framework=self.framework,
            model_path=self.model_path,
            task_type=self.task_type,
            feature_names=self.feature_names,
            metadata=self.metadata,
            dataset_name=self.dataset_name,
            train_script=self.train_script,
        )

    @property
    def fs_model_id(self) -> str:
        """Filesystem-safe model ID (replaces hyphens with underscores)."""
        return self.model_id.replace("-", "_")

    @property
    def has_named_features(self) -> bool:
        """Whether this model uses named features (vs raw tensor input)."""
        return len(self.feature_names) > 0
