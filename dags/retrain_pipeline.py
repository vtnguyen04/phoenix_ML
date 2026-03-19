# ruff: noqa: PLC0415
"""
Airflow DAG for Auto-Retraining Pipeline.
Triggered asynchronously via API when Drift is detected.
"""

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

def _train_model(**kwargs: Any) -> None:
    """Task 1: Generate reference features & Train new ML model"""
    # Import locally to avoid Airflow parse-time errors if dependencies are missing globally
    from scripts.train_model import train_and_export  # noqa: PLC0415
    
    timestamp = int(time.time())
    version = f"v{timestamp}"
    
    output_path = f"models/credit_risk/{version}/model.onnx"
    metrics_path = f"models/credit_risk/{version}/metrics.json"
    reference_path = f"models/credit_risk/{version}/reference_features.json"
    
    logger.info("Training new model version: %s", version)
    train_and_export(output_path, metrics_path=metrics_path, reference_path=reference_path)
    
    ti = kwargs['ti']
    ti.xcom_push(key='version', value=version)
    ti.xcom_push(key='output_path', value=output_path)
    ti.xcom_push(key='metrics_path', value=metrics_path)


def _log_mlflow(**kwargs: Any) -> None:
    """Task 2: Send metrics to MLflow Tracking Server"""
    import mlflow  # noqa: PLC0415
    
    ti = kwargs['ti']
    version = ti.xcom_pull(task_ids='train_model', key='version')
    metrics_path = ti.xcom_pull(task_ids='train_model', key='metrics_path')
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Using MLflow at %s", tracking_uri)
    
    mlflow.set_experiment("credit-risk-production")
    
    with open(metrics_path) as f:
        metrics = json.load(f)
        
    with mlflow.start_run(run_name=f"airflow-retrain-{version}"):
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_params({
            k: str(v) for k, v in metrics.items() 
            if not isinstance(v, (int, float)) and k != "all_features"
        })
    logger.info("Successfully logged metrics to MLflow")


def _register_postgres(**kwargs: Any) -> None:
    """Task 3: Register model into Postgres Database for inference routing"""
    from src.domain.inference.entities.model import Model  # noqa: PLC0415
    from src.infrastructure.persistence.database import get_db  # noqa: PLC0415
    from src.infrastructure.persistence.postgres_model_registry import (
        PostgresModelRegistry,  # noqa: PLC0415
    )
    
    ti = kwargs['ti']
    version = ti.xcom_pull(task_ids='train_model', key='version')
    output_path = ti.xcom_pull(task_ids='train_model', key='output_path')
    metrics_path = ti.xcom_pull(task_ids='train_model', key='metrics_path')
    
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
            logger.info("Successfully registered %s as challenger in Postgres", version)
            break

    asyncio.run(_async_register())


with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # Triggered manually or by API only
    catchup=False,
    tags=["auto-retrain", "mlops"],
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        provide_context=True,
    )

    mlflow_task = PythonOperator(
        task_id="log_mlflow",
        python_callable=_log_mlflow,
        provide_context=True,
    )

    register_task = PythonOperator(
        task_id="register_postgres",
        python_callable=_register_postgres,
        provide_context=True,
    )

    # Define DAG workflow
    train_task >> mlflow_task >> register_task
