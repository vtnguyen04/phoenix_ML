# ruff: noqa: PLC0415
"""
Self-Healing Pipeline DAG — Model-Agnostic Production Grade.

Triggered by FastAPI MonitoringService when drift is detected.
Executes the full remediation lifecycle:
  1. send_alert      — Notify via webhook
  2. rollback        — Archive challengers via API
  3. train_model     — Train new version using configured script
  4. log_mlflow      — Log metrics to MLflow
  5. register_model  — Register challenger via API

Architecture:
  - Uses model_configs/ YAML to resolve task-specific training scripts
  - All persistence via FastAPI REST API (no direct DB access)
  - Supports classification, regression, and any future task types
  - Model-agnostic: model_id drives all configuration
"""

import importlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("API_URL", "http://api:8000")

# Resolve project root — works in Docker, Airflow, and local dev
_PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
)


def _resolve_train_function(model_id: str) -> Any:
    """
    Resolve the training function for a given model_id.

    Looks up model_configs/{model_id}.yaml →  train_script field,
    then dynamically imports the train_and_export function.

    Falls back to examples.credit_risk.train for backward compatibility.
    """
    # Ensure project root is on sys.path for imports
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Try loading from model config YAML
    config_path = _PROJECT_ROOT / "model_configs" / f"{model_id}.yaml"
    if config_path.exists():
        try:
            import yaml  # type: ignore[import-untyped]

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            train_script = config.get("train_script", "")
            if train_script:
                # Convert path like "examples/house_price/train.py"
                # to module like "examples.house_price.train"
                module_path = train_script.replace("/", ".").removesuffix(".py")
                logger.info(
                    "Loading training module: %s (from %s)",
                    module_path,
                    config_path.name,
                )
                module = importlib.import_module(module_path)
                if hasattr(module, "train_and_export"):
                    return module.train_and_export
                # Try alternative function names
                for fn_name in ["train", "train_model", "main"]:
                    if hasattr(module, fn_name):
                        return getattr(module, fn_name)
                raise ImportError(f"No train_and_export/train/main function in {module_path}")
        except Exception as e:
            logger.warning(
                "Failed to load training from config %s: %s. "
                "Falling back to examples.credit_risk.train",
                config_path.name,
                e,
            )

    # Fallback: examples/credit_risk/train.py (backward compatibility)
    from examples.credit_risk.train import train_and_export

    return train_and_export


# ─── DAG Tasks ────────────────────────────────────────────────────


def _send_alert(**kwargs: Any) -> None:
    """Dispatch drift alert via webhook (Slack-compatible payload)."""
    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", "unknown")
    reason = conf.get("reason", "Drift detected")
    drift_score = conf.get("drift_score", 0.0)

    webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "")
    payload = {
        "text": f"*DRIFT ALERT* — model `{model_id}`",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Self-Healing Pipeline Triggered*\n"
                        f"Model: `{model_id}`\n"
                        f"Drift Score: `{drift_score:.4f}`\n"
                        f"Reason: {reason}\n"
                        f"Time: `{datetime.now(UTC).isoformat()}`"
                    ),
                },
            }
        ],
    }

    if webhook_url:
        try:
            resp = httpx.post(webhook_url, json=payload, timeout=5.0)
            logger.info("Alert sent to webhook (status=%d)", resp.status_code)
        except Exception as e:
            logger.warning("Webhook failed (non-blocking): %s", e)
    else:
        logger.info(
            "Alert (no webhook): model=%s drift=%.4f reason=%s",
            model_id,
            drift_score,
            reason,
        )


def _rollback_challenger(**kwargs: Any) -> None:
    """Archive challengers via FastAPI — delegates to PostgresModelRegistry."""
    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", os.environ.get("DEFAULT_MODEL_ID", "credit-risk"))

    resp = httpx.post(
        f"{_API_URL}/models/rollback",
        json={"model_id": model_id},
        timeout=10.0,
    )
    resp.raise_for_status()
    result = resp.json()
    logger.info(
        "Rollback complete: champion=%s archived=%s",
        result.get("champion"),
        result.get("archived_challengers"),
    )


def _train_model(**kwargs: Any) -> None:
    """
    Train a new ML model version — model-agnostic.

    Resolves training function AND data_path from model_configs/{model_id}.yaml.
    The framework auto-detects the user's config to find the correct
    training script and dataset location.
    """
    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", os.environ.get("DEFAULT_MODEL_ID", "credit-risk"))
    fs_model_id = model_id.replace("-", "_")

    timestamp = int(time.time())
    version = f"v{timestamp}"

    output_path = str(_PROJECT_ROOT / "models" / fs_model_id / version / "model.onnx")
    metrics_path = str(_PROJECT_ROOT / "models" / fs_model_id / version / "metrics.json")
    reference_path = str(
        _PROJECT_ROOT / "models" / fs_model_id / version / "reference_features.json"
    )

    # Resolve data_path from model config via ModelConfigLoader
    data_path: str | None = None
    config_path = _PROJECT_ROOT / "model_configs" / f"{model_id}.yaml"
    if config_path.exists():
        try:
            from phoenix_ml.infrastructure.bootstrap.model_config_loader import (
                load_model_config,
            )

            model_config = load_model_config(config_path)
            if model_config.data_path:
                resolved = _PROJECT_ROOT / model_config.data_path
                data_path = str(resolved)
                logger.info("Resolved data_path: %s", data_path)
        except Exception as e:
            logger.warning("Failed to read config for %s: %s", model_id, e)

    # Dynamically resolve training function
    train_fn = _resolve_train_function(model_id)

    logger.info(
        "Training model %s version %s using %s (data=%s)",
        model_id,
        version,
        train_fn.__module__,
        data_path or "default",
    )

    # Call with supported kwargs (different scripts accept different args)
    import inspect

    sig = inspect.signature(train_fn)
    call_kwargs: dict[str, str] = {}
    if "metrics_path" in sig.parameters:
        call_kwargs["metrics_path"] = metrics_path
    if "reference_path" in sig.parameters:
        call_kwargs["reference_path"] = reference_path
    if "data_path" in sig.parameters and data_path:
        call_kwargs["data_path"] = data_path

    train_fn(output_path, **call_kwargs)

    ti = kwargs["ti"]
    ti.xcom_push(key="version", value=version)
    ti.xcom_push(key="output_path", value=output_path)
    ti.xcom_push(key="metrics_path", value=metrics_path)
    ti.xcom_push(key="model_id", value=model_id)


def _log_mlflow(**kwargs: Any) -> None:
    """
    Log training metrics to MLflow — model-agnostic.

    Automatically filters numeric metrics and string params.
    Works with any task type (classification, regression, etc.)
    """
    import mlflow

    ti = kwargs["ti"]
    version = ti.xcom_pull(task_ids="train_model", key="version")
    metrics_path = ti.xcom_pull(task_ids="train_model", key="metrics_path")

    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", os.environ.get("DEFAULT_MODEL_ID", "credit-risk"))

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{model_id}-production")

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Split into numeric metrics and string params (model-agnostic)
    numeric_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}
    string_params = {k: str(v) for k, v in all_metrics.items() if isinstance(v, str)}
    # Add task metadata
    string_params["model_id"] = model_id
    string_params["version"] = version

    with mlflow.start_run(run_name=f"self-healing-{version}"):
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_params(string_params)
        # Log the metrics JSON as artifact
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

    logger.info(
        "Metrics logged to MLflow for %s:%s (%d metrics, %d params)",
        model_id,
        version,
        len(numeric_metrics),
        len(string_params),
    )


def _register_model(**kwargs: Any) -> None:
    """Register new challenger via FastAPI — delegates to PostgresModelRegistry."""
    ti = kwargs["ti"]
    version = ti.xcom_pull(task_ids="train_model", key="version")
    output_path = ti.xcom_pull(task_ids="train_model", key="output_path")
    metrics_path = ti.xcom_pull(task_ids="train_model", key="metrics_path")

    with open(metrics_path) as f:
        metrics = json.load(f)

    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", os.environ.get("DEFAULT_MODEL_ID", "credit-risk"))

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
    result = resp.json()
    logger.info(
        "Registered %s:%s as %s",
        model_id,
        result["version"],
        result["stage"],
    )


# ─── DAG Definition ──────────────────────────────────────────────


with DAG(
    dag_id="self_healing_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["self-healing", "auto-retrain", "mlops"],
    doc_md=__doc__,
) as dag:
    alert_task = PythonOperator(
        task_id="send_alert",
        python_callable=_send_alert,
    )
    rollback_task = PythonOperator(
        task_id="rollback_challenger",
        python_callable=_rollback_challenger,
    )
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )
    mlflow_task = PythonOperator(
        task_id="log_mlflow",
        python_callable=_log_mlflow,
    )
    register_task = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    alert_task >> rollback_task >> train_task >> mlflow_task >> register_task
