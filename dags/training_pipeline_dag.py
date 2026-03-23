# ruff: noqa: PLC0415
"""Dynamic DAG factory for per-model training pipelines.

Reads YAML configs from ``model_configs/*.yaml`` at DAG parse time and
registers one Airflow DAG per config into ``globals()``.

Config path:
    ``$PROJECT_ROOT/model_configs/<model_id>.yaml``

Expected YAML schema::

    model_id: str
    version: str
    train_script: str          # dotted module path to train_and_export()
    data_path: str             # path to CSV training data
    task_type: str             # "classification" | "regression"
    retrain_trigger: str       # "manual" | "daily" | "weekly" | "data_change"

DAG steps:
    validate_data → train_model → evaluate_model → log_to_mlflow → register_model

Side effects:
    - Registers DAGs as ``training_<model_id>`` in module globals.
    - Each DAG creates MLflow runs and calls the Phoenix API on register.

Error handling:
    - Missing or malformed configs are logged and skipped.
    - Individual task failures do not prevent DAG registration.
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("API_URL", "http://api:8000")
_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

_PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
)


def _get_model_configs() -> dict[str, dict[str, Any]]:
    """Load all model configs from model_configs/ directory."""
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    config_dir = _PROJECT_ROOT / "model_configs"
    if not config_dir.exists():
        return {}

    try:
        from phoenix_ml.infrastructure.bootstrap.model_config_loader import (
            load_all_model_configs,
        )

        configs = load_all_model_configs(config_dir)
        return {
            model_id: {
                "model_id": model_id,
                "version": cfg.version,
                "train_script": cfg.train_script,
                "data_path": cfg.data_path,
                "task_type": cfg.task_type,
                "retrain_trigger": cfg.retrain_trigger,
                "feature_names": cfg.feature_names,
                "model_path": str(cfg.model_path) if cfg.model_path else None,
            }
            for model_id, cfg in configs.items()
        }
    except Exception as e:
        logger.warning("Failed to load model configs: %s", e)
        return {}


# ── Task Functions ───────────────────────────────────────────────


def validate_data(model_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Validate training data quality before training."""
    data_path = model_config.get("data_path", "")
    model_id = model_config["model_id"]

    result = {
        "model_id": model_id,
        "data_path": data_path,
        "validation_passed": True,
        "errors": [],
    }

    if data_path and Path(data_path).exists():
        import pandas as pd

        df = pd.read_csv(data_path)
        result["row_count"] = len(df)
        result["column_count"] = len(df.columns)
        result["null_percent"] = round(df.isnull().sum().sum() / df.size * 100, 2)

        if len(df) < 10:
            result["validation_passed"] = False
            result["errors"].append(f"Insufficient data: {len(df)} rows")

        if result["null_percent"] > 50:
            result["validation_passed"] = False
            result["errors"].append(f"Too many nulls: {result['null_percent']}%")
    else:
        logger.warning("Data path not found: %s, skipping validation", data_path)

    logger.info("Data validation for %s: %s", model_id, result)
    return result


def train_model(model_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Train model using the configured train_script."""
    model_id = model_config["model_id"]
    version = model_config.get("version", "v1")
    train_script = model_config.get("train_script", "")

    logger.info("🏋️ Training %s:%s using %s", model_id, version, train_script)

    fs_model_id = model_id.replace("-", "_")
    output_path = str(_PROJECT_ROOT / "models" / fs_model_id / version / "model.onnx")
    metrics_path = str(_PROJECT_ROOT / "models" / fs_model_id / version / "metrics.json")

    result = {
        "model_id": model_id,
        "version": version,
        "output_path": output_path,
        "metrics_path": metrics_path,
        "success": False,
    }

    if train_script:
        try:
            import importlib

            module_path = train_script.replace("/", ".").removesuffix(".py")
            module = importlib.import_module(module_path)
            train_fn = module.train_and_export
            train_fn(
                output_path=output_path,
                metrics_path=metrics_path,
                data_path=model_config.get("data_path", ""),
            )
            result["success"] = True
            logger.info("✅ Training complete: %s", output_path)
        except Exception as e:
            result["error"] = str(e)
            logger.error("❌ Training failed for %s: %s", model_id, e)
    else:
        result["error"] = "No train_script configured"

    return result


def evaluate_model(model_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Load and evaluate training metrics against thresholds."""
    model_id = model_config["model_id"]
    version = model_config.get("version", "v1")
    fs_model_id = model_id.replace("-", "_")

    metrics_path = _PROJECT_ROOT / "models" / fs_model_id / version / "metrics.json"

    result = {"model_id": model_id, "passed": False, "metrics": {}}

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        result["metrics"] = metrics

        # Check minimum thresholds
        min_accuracy = 0.5  # configurable per model
        accuracy = metrics.get("accuracy", 0)
        if accuracy >= min_accuracy:
            result["passed"] = True
            logger.info("✅ Evaluation passed for %s: accuracy=%.4f", model_id, accuracy)
        else:
            logger.warning(
                "❌ Evaluation failed for %s: accuracy=%.4f < %.4f",
                model_id, accuracy, min_accuracy,
            )
    else:
        result["error"] = f"Metrics file not found: {metrics_path}"

    return result


def log_to_mlflow(model_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Log training results to MLflow."""
    model_id = model_config["model_id"]
    version = model_config.get("version", "v1")
    fs_model_id = model_id.replace("-", "_")

    metrics_path = _PROJECT_ROOT / "models" / fs_model_id / version / "metrics.json"
    model_path = _PROJECT_ROOT / "models" / fs_model_id / version / "model.onnx"

    result = {"model_id": model_id, "mlflow_logged": False}

    try:
        import mlflow

        mlflow.set_tracking_uri(_MLFLOW_URI)
        mlflow.set_experiment(f"phoenix-{model_id}")

        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        with mlflow.start_run(run_name=f"{model_id}-{version}-airflow"):
            mlflow.log_params({
                "model_id": model_id,
                "version": version,
                "task_type": model_config.get("task_type", "classification"),
                "source": "airflow-pipeline",
            })

            # Log all numeric metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Log model artifact
            if model_path.exists():
                mlflow.log_artifact(str(model_path))

            # Log metrics JSON
            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path))

        result["mlflow_logged"] = True
        result["metrics"] = metrics
        logger.info("✅ MLflow logged for %s:%s", model_id, version)
    except Exception as e:
        result["error"] = str(e)
        logger.error("❌ MLflow logging failed for %s: %s", model_id, e)

    return result


def register_model(model_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Register trained model as challenger via API."""
    model_id = model_config["model_id"]
    version = model_config.get("version", "v1")

    result = {"model_id": model_id, "registered": False}

    try:
        response = httpx.post(
            f"{_API_URL}/models/{model_id}/versions",
            json={"version": version, "strategy": "canary"},
            timeout=30,
        )
        if response.status_code < 300:
            result["registered"] = True
            logger.info("✅ Model registered: %s:%s", model_id, version)
        else:
            result["error"] = f"API returned {response.status_code}"
    except Exception as e:
        result["error"] = str(e)
        logger.warning("Registration failed (non-critical): %s", e)

    return result


# ── DAG Factory ──────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "phoenix-ml",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

model_configs = _get_model_configs()

for _model_id, _config in model_configs.items():
    dag_id = f"training_{_model_id.replace('-', '_')}"
    schedule = _config.get("retrain_trigger", "manual")

    # Map trigger to Airflow schedule
    schedule_interval = None
    if schedule == "daily":
        schedule_interval = "@daily"
    elif schedule == "weekly":
        schedule_interval = "@weekly"
    elif schedule == "data_change":
        schedule_interval = timedelta(hours=6)
    # manual or unknown → None (manual trigger)

    dag = DAG(
        dag_id=dag_id,
        default_args=DEFAULT_ARGS,
        description=f"Training pipeline for {_model_id}",
        schedule_interval=schedule_interval,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        catchup=False,
        tags=["phoenix-ml", "training", _config.get("task_type", "classification")],
    )

    # Capture config in closure
    def _make_task_fn(fn, config):
        def wrapper(**kwargs):
            return fn(config, **kwargs)
        return wrapper

    t1 = PythonOperator(
        task_id="validate_data",
        python_callable=_make_task_fn(validate_data, _config),
        dag=dag,
    )

    t2 = PythonOperator(
        task_id="train_model",
        python_callable=_make_task_fn(train_model, _config),
        dag=dag,
    )

    t3 = PythonOperator(
        task_id="evaluate_model",
        python_callable=_make_task_fn(evaluate_model, _config),
        dag=dag,
    )

    t4 = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=_make_task_fn(log_to_mlflow, _config),
        dag=dag,
    )

    t5 = PythonOperator(
        task_id="register_model",
        python_callable=_make_task_fn(register_model, _config),
        dag=dag,
    )

    t1 >> t2 >> t3 >> t4 >> t5  # type: ignore[operator]

    # Register in globals for Airflow to discover
    globals()[dag_id] = dag
