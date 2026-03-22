"""
ModelConfigLoader — Load model configurations from YAML files.

Reads model config YAML files from MODEL_CONFIG_DIR, enabling users to
define models for any ML problem type without modifying framework code.

Example YAML (model_configs/credit-risk.yaml):
    model_id: credit-risk
    version: v1
    framework: onnx
    task_type: classification
    model_path: models/credit_risk/v1/model.onnx
    train_script: examples/credit_risk/train.py
    dataset_name: german-credit-openml
    feature_names:
      - duration
      - credit_amount
      - ...
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.domain.inference.value_objects.model_config import ModelConfig

logger = logging.getLogger(__name__)

# Use yaml if available, fall back to JSON-based config
try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


def _parse_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file. Requires PyYAML."""
    if not _HAS_YAML:
        msg = f"PyYAML required to load {path}. Install: pip install pyyaml"
        raise ImportError(msg)
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f) or {}
        return result


def _parse_json(path: Path) -> dict[str, Any]:
    """Parse a JSON file."""
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _dict_to_model_config(data: dict[str, Any]) -> ModelConfig:
    """Convert a raw dict to a ModelConfig value object."""
    feature_names = data.get("feature_names", [])
    if isinstance(feature_names, list):
        feature_names = tuple(feature_names)

    raw_metadata = data.get("metadata", {})
    if isinstance(raw_metadata, dict):
        metadata_tuple = tuple(raw_metadata.items())
    else:
        metadata_tuple = ()

    # Parse monitoring section (per-model drift & metric config)
    monitoring = data.get("monitoring", {})
    if not isinstance(monitoring, dict):
        monitoring = {}

    # Parse data_source section
    data_source = data.get("data_source", {})
    if not isinstance(data_source, dict):
        data_source = {}

    # Parse retrain section
    retrain = data.get("retrain", {})
    if not isinstance(retrain, dict):
        retrain = {}

    # Task-type defaults mapping (OCP: add new task types via dict entry)
    # Format: (drift_test, primary_metric, default_data_source, default_trigger, drift_enabled)
    _TASK_DEFAULTS: dict[str, tuple[str, str, str, str, bool]] = {
        "classification": ("ks", "accuracy", "file", "drift", True),
        "regression": ("wasserstein", "rmse", "file", "drift", True),
        "image": ("chi2", "accuracy", "file", "drift", True),
        "image_classification": ("chi2", "accuracy", "file", "drift", True),
        "object_detection": ("ks", "map", "dvc", "data_change", False),
        "nlp": ("ks", "accuracy", "file", "manual", False),
        "custom": ("ks", "accuracy", "file", "manual", True),
    }
    task_type = data.get("task_type", "classification")
    defaults = _TASK_DEFAULTS.get(task_type, ("ks", "accuracy", "file", "drift", True))
    (
        default_drift_test,
        default_primary_metric,
        default_ds_type,
        default_trigger,
        default_drift_enabled,
    ) = defaults

    return ModelConfig(
        model_id=data.get("model_id", ""),
        version=data.get("version", "v1"),
        framework=data.get("framework", "onnx"),
        model_path=data.get("model_path", ""),
        task_type=task_type,
        feature_names=feature_names,
        metadata=metadata_tuple,
        dataset_name=data.get("dataset_name", ""),
        data_path=data.get("data_path", ""),
        train_script=data.get("train_script", ""),
        monitoring_drift_test=monitoring.get("drift_test", default_drift_test),
        monitoring_primary_metric=monitoring.get("primary_metric", default_primary_metric),
        data_source_type=data_source.get("type", default_ds_type),
        data_source_query=data_source.get("query", ""),
        data_source_connection=data_source.get("connection", ""),
        retrain_trigger=retrain.get("trigger", default_trigger),
        retrain_schedule=retrain.get("schedule", ""),
        drift_detection_enabled=retrain.get("drift_detection", default_drift_enabled),
    )


def load_model_config(config_path: Path) -> ModelConfig:
    """Load a single model config from a YAML or JSON file.

    Args:
        config_path: Absolute or relative path to the config file.

    Returns:
        Parsed ModelConfig value object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file format is unsupported.
    """
    if not config_path.exists():
        msg = f"Model config not found: {config_path}"
        raise FileNotFoundError(msg)

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _parse_yaml(config_path)
    elif suffix == ".json":
        data = _parse_json(config_path)
    else:
        msg = f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json"
        raise ValueError(msg)

    return _dict_to_model_config(data)


def load_all_model_configs(config_dir: Path) -> dict[str, ModelConfig]:
    """Load all model configs from a directory.

    Scans for .yaml, .yml, and .json files in the config directory.

    Args:
        config_dir: Path to the directory containing model config files.

    Returns:
        Dict mapping model_id to ModelConfig.
    """
    configs: dict[str, ModelConfig] = {}
    if not config_dir.exists():
        logger.info("Model config directory not found: %s — starting without configs", config_dir)
        return configs

    extensions = ("*.yaml", "*.yml", "*.json")
    for ext in extensions:
        for path in sorted(config_dir.glob(ext)):
            try:
                config = load_model_config(path)
                if config.model_id:
                    configs[config.model_id] = config
                    logger.info("📋 Loaded model config: %s (%s)", config.model_id, path.name)
                else:
                    logger.warning("⚠️ Skipping config without model_id: %s", path)
            except Exception as e:
                logger.warning("⚠️ Failed to load model config %s: %s", path, e)

    return configs


def load_features_from_metrics(metrics_path: Path) -> list[str]:
    """Extract feature names from a training metrics.json file.

    The metrics.json produced by training scripts contains an
    'all_features' key with the ordered feature names.

    Args:
        metrics_path: Path to the metrics.json file.

    Returns:
        List of feature name strings, empty if not found.
    """
    if not metrics_path.exists():
        return []
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        features = data.get("all_features", [])
        if isinstance(features, list):
            return [str(f) for f in features]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("⚠️ Failed to read features from %s: %s", metrics_path, e)
    return []
