"""Immutable value object describing a model's configuration.

Loaded from YAML files in ``MODEL_CONFIG_DIR``. Each model's task type,
features, training script, and monitoring settings are captured here.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Frozen dataclass holding all model configuration fields.

    All fields default to safe zero-values. Loaded from external YAML
    by ``model_config_loader.load_model_config()``.

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
        data_path: Relative path to training dataset (CSV, NPZ, directory).
        train_script: Path to training script for self-healing retrain.
        monitoring_drift_test: Statistical test for drift detection.
            Supported: "ks", "psi", "wasserstein", "chi2".
        monitoring_primary_metric: Primary evaluation metric name.
            E.g. "accuracy" for classification, "rmse" for regression.
        data_source_type: Where training data comes from.
            "file" — CSV/NPZ on disk (default).
            "database" — SQL query from a database.
            "dvc" — DVC-versioned large datasets (images, etc.).
        data_source_query: SQL query for database-sourced data.
        data_source_connection: Database connection name for database sources.
        retrain_trigger: What triggers model retraining.
            "drift" — auto-retrain when drift is detected (default).
            "manual" — user triggers via API or Airflow UI.
            "data_change" — retrain when DVC data changes.
            "scheduled" — cron-based periodic retrain.
        retrain_schedule: Cron expression for scheduled retraining.
        drift_detection_enabled: Whether drift monitoring is active.
            Set to False for tasks without meaningful drift detection
            (e.g. object detection, NLP).
    """

    model_id: str
    version: str = "v1"
    framework: str = "onnx"
    model_path: str = ""
    task_type: str = "classification"
    feature_names: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    dataset_name: str = ""
    data_path: str = ""
    train_script: str = ""
    monitoring_drift_test: str = "ks"
    monitoring_primary_metric: str = "accuracy"

    # Data source configuration
    data_source_type: str = "file"
    data_source_query: str = ""
    data_source_connection: str = ""

    # Retrain trigger configuration
    retrain_trigger: str = "drift"
    retrain_schedule: str = ""
    drift_detection_enabled: bool = True

    # Optional pipeline steps (omit for default train → validate → register)
    pipeline_steps: tuple[tuple[str, ...], ...] = ()

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
            data_path=self.data_path,
            train_script=self.train_script,
            monitoring_drift_test=self.monitoring_drift_test,
            monitoring_primary_metric=self.monitoring_primary_metric,
            data_source_type=self.data_source_type,
            data_source_query=self.data_source_query,
            data_source_connection=self.data_source_connection,
            retrain_trigger=self.retrain_trigger,
            retrain_schedule=self.retrain_schedule,
            drift_detection_enabled=self.drift_detection_enabled,
        )

    @property
    def fs_model_id(self) -> str:
        """Filesystem-safe model ID (replaces hyphens with underscores)."""
        return self.model_id.replace("-", "_")

    @property
    def has_named_features(self) -> bool:
        """Whether this model uses named features (vs raw tensor input)."""
        return len(self.feature_names) > 0

    @property
    def uses_dvc(self) -> bool:
        """Whether this model's data is versioned by DVC."""
        return self.data_source_type == "dvc"

    @property
    def is_drift_monitored(self) -> bool:
        """Whether drift detection should run for this model."""
        return self.drift_detection_enabled and self.retrain_trigger == "drift"
