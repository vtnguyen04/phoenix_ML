# ruff: noqa: PLC0415
"""
Self-Healing Pipeline DAG — Production Grade.

Triggered by FastAPI MonitoringService when drift is detected.
Executes the full remediation lifecycle:
  1. send_alert      — Notify via webhook (Slack/Discord)
  2. rollback        — Archive active challengers, keep champion serving
  3. train_model     — Train a new model version
  4. log_mlflow      — Log metrics/params to MLflow tracking server
  5. register_model  — Register new model as challenger in Postgres

Safety: max_active_runs=1 prevents concurrent pipeline executions.
"""

import asyncio
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

# ---------------------------------------------------------------------------
# Task 1: Send Alert
# ---------------------------------------------------------------------------

def _send_alert(**kwargs: Any) -> None:
    """Dispatch drift alert via webhook (Slack-compatible payload)."""
    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", "unknown")
    reason = conf.get("reason", "Drift detected")
    drift_score = conf.get("drift_score", 0.0)

    webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "")
    payload = {
        "text": f"🚨 *DRIFT ALERT* — model `{model_id}`",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"🚨 *Self-Healing Pipeline Triggered*\n"
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
            logger.info("✅ Alert sent to webhook (status=%d)", resp.status_code)
        except Exception as e:
            logger.warning("⚠️ Webhook failed (non-blocking): %s", e)
    else:
        logger.info("📢 Alert (no webhook configured): %s — %s", model_id, reason)


# ---------------------------------------------------------------------------
# Task 2: Rollback Challenger
# ---------------------------------------------------------------------------

def _rollback_challenger(**kwargs: Any) -> None:
    """Archive any active challenger models so champion keeps serving."""
    from src.infrastructure.persistence.database import get_db
    from src.infrastructure.persistence.postgres_model_registry import (
        PostgresModelRegistry,
    )

    conf = kwargs.get("dag_run", {}).conf or {}
    model_id = conf.get("model_id", "credit-risk")

    async def _async_rollback() -> None:
        async for db in get_db():
            repo = PostgresModelRegistry(db)
            models = await repo.get_active_versions(model_id)

            champion = None
            challengers = []
            for m in models:
                role = m.metadata.get("role", "")
                if role == "champion":
                    champion = m
                elif role == "challenger":
                    challengers.append(m)

            if not champion:
                logger.warning("⚠️ No champion for %s — skip rollback", model_id)
                return

            if not challengers:
                logger.info("ℹ️ No challengers to rollback for %s", model_id)
                return

            for c in challengers:
                await repo.update_stage(model_id, c.version, "archived")
                logger.info(
                    "⏪ ROLLBACK: %s challenger %s → archived, champion %s active",
                    model_id,
                    c.version,
                    champion.version,
                )
            await db.commit()
            break

    asyncio.run(_async_rollback())


# ---------------------------------------------------------------------------
# Task 3: Train New Model
# ---------------------------------------------------------------------------

def _train_model(**kwargs: Any) -> None:
    """Train a new ML model version and export as ONNX."""
    from scripts.train_model import train_and_export

    timestamp = int(time.time())
    version = f"v{timestamp}"

    output_path = f"models/credit_risk/{version}/model.onnx"
    metrics_path = f"models/credit_risk/{version}/metrics.json"
    reference_path = f"models/credit_risk/{version}/reference_features.json"

    logger.info("🏋️ Training new model version: %s", version)
    train_and_export(
        output_path,
        metrics_path=metrics_path,
        reference_path=reference_path,
    )

    ti = kwargs["ti"]
    ti.xcom_push(key="version", value=version)
    ti.xcom_push(key="output_path", value=output_path)
    ti.xcom_push(key="metrics_path", value=metrics_path)


# ---------------------------------------------------------------------------
# Task 4: Log to MLflow
# ---------------------------------------------------------------------------

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
    logger.info("📈 Metrics logged to MLflow for %s", version)


# ---------------------------------------------------------------------------
# Task 5: Register Challenger in Postgres
# ---------------------------------------------------------------------------

def _register_model(**kwargs: Any) -> None:
    """Register the newly trained model as a challenger in Postgres."""
    from src.domain.inference.entities.model import Model
    from src.infrastructure.persistence.database import get_db
    from src.infrastructure.persistence.postgres_model_registry import (
        PostgresModelRegistry,
    )

    ti = kwargs["ti"]
    version = ti.xcom_pull(task_ids="train_model", key="version")
    output_path = ti.xcom_pull(task_ids="train_model", key="output_path")
    metrics_path = ti.xcom_pull(task_ids="train_model", key="metrics_path")

    with open(metrics_path) as f:
        metrics = json.load(f)

    async def _async_register() -> None:
        async for db in get_db():
            repo = PostgresModelRegistry(db)
            new_model = Model(
                id="credit-risk",
                version=version,
                uri=f"local:///{output_path}",
                framework="onnx",
                metadata={"metrics": metrics, "role": "challenger"},
                created_at=datetime.now(UTC),
                is_active=True,
            )
            await repo.save(new_model)
            await db.commit()
            logger.info(
                "🗄️ Registered %s as challenger in Postgres", version
            )
            break

    asyncio.run(_async_register())


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

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

    # Self-healing pipeline flow
    alert_task >> rollback_task >> train_task >> mlflow_task >> register_task
