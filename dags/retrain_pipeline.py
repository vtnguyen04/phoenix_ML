# ruff: noqa: PLC0415
"""
Self-Healing Pipeline DAG — Production Grade.

Triggered by FastAPI MonitoringService when drift is detected.
Executes the full remediation lifecycle:
  1. send_alert      — Notify via webhook
  2. rollback        — Archive challengers via API
  3. train_model     — Train a new model version
  4. log_mlflow      — Log metrics to MLflow
  5. register_model  — Register challenger via API

All persistence operations go through the FastAPI REST API,
which uses the repository pattern internally. The DAG only
orchestrates HTTP calls — no direct database access.
"""

import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

_API_URL = os.environ.get("API_URL", "http://api:8000")


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
    model_id = conf.get("model_id", "credit-risk")

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
    """Train a new ML model version and export as ONNX."""
    from scripts.train_model import train_and_export

    timestamp = int(time.time())
    version = f"v{timestamp}"

    output_path = f"models/credit_risk/{version}/model.onnx"
    metrics_path = f"models/credit_risk/{version}/metrics.json"
    reference_path = f"models/credit_risk/{version}/reference_features.json"

    logger.info("Training new model version: %s", version)
    train_and_export(
        output_path,
        metrics_path=metrics_path,
        reference_path=reference_path,
    )

    ti = kwargs["ti"]
    ti.xcom_push(key="version", value=version)
    ti.xcom_push(key="output_path", value=output_path)
    ti.xcom_push(key="metrics_path", value=metrics_path)


def _log_mlflow(**kwargs: Any) -> None:
    """Send training metrics to MLflow Tracking Server."""
    import mlflow

    ti = kwargs["ti"]
    version = ti.xcom_pull(task_ids="train_model", key="version")
    metrics_path = ti.xcom_pull(task_ids="train_model", key="metrics_path")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-risk-production")

    with open(metrics_path) as f:
        metrics = json.load(f)

    with mlflow.start_run(run_name=f"self-healing-{version}"):
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.log_params(
            {
                k: str(v)
                for k, v in metrics.items()
                if not isinstance(v, (int, float)) and k != "all_features"
            }
        )
    logger.info("Metrics logged to MLflow for %s", version)


def _register_model(**kwargs: Any) -> None:
    """Register new challenger via FastAPI — delegates to PostgresModelRegistry."""
    ti = kwargs["ti"]
    version = ti.xcom_pull(task_ids="train_model", key="version")
    output_path = ti.xcom_pull(task_ids="train_model", key="output_path")
    metrics_path = ti.xcom_pull(task_ids="train_model", key="metrics_path")

    with open(metrics_path) as f:
        metrics = json.load(f)

    resp = httpx.post(
        f"{_API_URL}/models/register",
        json={
            "model_id": "credit-risk",
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
    logger.info("Registered %s as %s", result["version"], result["stage"])


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
