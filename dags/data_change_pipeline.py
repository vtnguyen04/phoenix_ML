# ruff: noqa: PLC0415
"""
Data-Change Pipeline DAG — Triggered when DVC-tracked data is updated.

Complement to self_healing_pipeline (drift-based). This DAG is for models
whose retrain trigger is "data_change" (e.g., object detection, NLP).

When users add/update data via DVC (dvc add → dvc push), this pipeline:
  1. detect_data_change — Check `dvc status` for modified data
  2. train_model         — Train new model version
  3. log_mlflow          — Log metrics to MLflow
  4. register_model      — Register challenger via API

Architecture:
  - Scheduled to check for data changes periodically (every 6 hours)
  - Only trains if DVC status shows actual changes
  - Reads model_configs/ to find models with retrain.trigger: data_change
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("API_URL", "http://api:8000")

_PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
)


def _get_data_change_models() -> list[dict[str, Any]]:
    """Find all models configured with retrain.trigger: data_change."""
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    config_dir = _PROJECT_ROOT / "model_configs"
    if not config_dir.exists():
        return []

    models = []
    try:
        from phoenix_ml.infrastructure.bootstrap.model_config_loader import (
            load_all_model_configs,
        )

        configs = load_all_model_configs(config_dir)
        for model_id, config in configs.items():
            if config.retrain_trigger == "data_change":
                models.append(
                    {
                        "model_id": model_id,
                        "data_path": config.data_path,
                        "data_source_type": config.data_source_type,
                        "train_script": config.train_script,
                    }
                )
    except Exception as e:
        logger.warning("Failed to load model configs: %s", e)

    return models


def _detect_data_change(**kwargs: Any) -> bool:
    """
    Check if DVC-tracked data has changed.

    Returns True if changes detected (pipeline continues),
    False if no changes (pipeline short-circuits).
    """
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=30,
            check=False,
        )
        has_changes = bool(result.stdout.strip()) and "no changes" not in result.stdout.lower()

        if has_changes:
            logger.info("DVC data changes detected:\n%s", result.stdout)
            # Find which models are affected
            changed_models = _get_data_change_models()
            if changed_models:
                ti = kwargs["ti"]
                ti.xcom_push(key="changed_models", value=changed_models)
                logger.info("Models to retrain: %s", [m["model_id"] for m in changed_models])
                return True
            logger.info("No data_change models configured — skipping")
            return False

        logger.info("No DVC data changes detected — skipping pipeline")
        return False
    except FileNotFoundError:
        logger.warning("DVC not installed — skipping data change detection")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("DVC status timed out — skipping")
        return False


def _train_changed_models(**kwargs: Any) -> None:
    """Train all models affected by data changes."""
    import importlib

    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    ti = kwargs["ti"]
    changed_models = ti.xcom_pull(task_ids="detect_data_change", key="changed_models") or []

    for model_info in changed_models:
        model_id = model_info["model_id"]
        fs_model_id = model_id.replace("-", "_")
        timestamp = int(time.time())
        version = f"v{timestamp}"

        output_path = str(_PROJECT_ROOT / "models" / fs_model_id / version / "model.onnx")
        metrics_path = str(_PROJECT_ROOT / "models" / fs_model_id / version / "metrics.json")
        reference_path = str(
            _PROJECT_ROOT / "models" / fs_model_id / version / "reference_features.json"
        )

        # Resolve training function
        train_script = model_info.get("train_script", "")
        if train_script:
            module_path = train_script.replace("/", ".").removesuffix(".py")
            module = importlib.import_module(module_path)
            train_fn = getattr(module, "train_and_export", None)
            if train_fn is None:
                for fn_name in ["train", "train_model", "main"]:
                    if hasattr(module, fn_name):
                        train_fn = getattr(module, fn_name)
                        break
            if train_fn is None:
                logger.error("No training function found for %s", model_id)
                continue
        else:
            logger.warning("No train_script for %s — skipping", model_id)
            continue

        logger.info("Training %s version %s (data_change trigger)", model_id, version)

        import inspect

        sig = inspect.signature(train_fn)
        call_kwargs: dict[str, str] = {}
        if "metrics_path" in sig.parameters:
            call_kwargs["metrics_path"] = metrics_path
        if "reference_path" in sig.parameters:
            call_kwargs["reference_path"] = reference_path

        data_path = model_info.get("data_path", "")
        if "data_path" in sig.parameters and data_path:
            call_kwargs["data_path"] = str(_PROJECT_ROOT / data_path)

        train_fn(output_path, **call_kwargs)

        ti.xcom_push(key=f"{model_id}_version", value=version)
        ti.xcom_push(key=f"{model_id}_output_path", value=output_path)
        ti.xcom_push(key=f"{model_id}_metrics_path", value=metrics_path)

    ti.xcom_push(key="trained_model_ids", value=[m["model_id"] for m in changed_models])


def _log_and_register(**kwargs: Any) -> None:
    """Log metrics to MLflow and register models via API."""
    import mlflow

    ti = kwargs["ti"]
    model_ids = ti.xcom_pull(task_ids="train_changed_models", key="trained_model_ids") or []

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    for model_id in model_ids:
        version = ti.xcom_pull(task_ids="train_changed_models", key=f"{model_id}_version")
        output_path = ti.xcom_pull(task_ids="train_changed_models", key=f"{model_id}_output_path")
        metrics_path = ti.xcom_pull(
            task_ids="train_changed_models", key=f"{model_id}_metrics_path"
        )

        if not version or not metrics_path:
            continue

        # Log to MLflow
        mlflow.set_experiment(f"{model_id}-production")
        with open(metrics_path) as f:
            all_metrics = json.load(f)

        numeric_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}
        string_params = {k: str(v) for k, v in all_metrics.items() if isinstance(v, str)}
        string_params["model_id"] = model_id
        string_params["version"] = version
        string_params["trigger"] = "data_change"

        with mlflow.start_run(run_name=f"data-change-{version}"):
            mlflow.log_metrics(numeric_metrics)
            mlflow.log_params(string_params)
            mlflow.log_artifact(metrics_path, artifact_path="metrics")

        # Register via API
        with open(metrics_path) as f:
            metrics = json.load(f)

        resp = httpx.post(
            f"{_API_URL}/models/register",
            json={
                "model_id": model_id,
                "version": version,
                "uri": f"local:///{output_path}",
                "framework": "onnx",
                "stage": "challenger",
                "metrics": metrics,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        logger.info("Registered %s:%s from data_change trigger", model_id, version)


# ─── DAG Definition ──────────────────────────────────────────────

with DAG(
    dag_id="data_change_pipeline",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule="0 */6 * * *",  # Check every 6 hours
    catchup=False,
    max_active_runs=1,
    tags=["data-change", "auto-retrain", "dvc", "mlops"],
    doc_md=__doc__,
) as dag:
    detect_task = ShortCircuitOperator(
        task_id="detect_data_change",
        python_callable=_detect_data_change,
    )
    train_task = PythonOperator(
        task_id="train_changed_models",
        python_callable=_train_changed_models,
    )
    register_task = PythonOperator(
        task_id="log_and_register",
        python_callable=_log_and_register,
    )

    detect_task >> train_task >> register_task
