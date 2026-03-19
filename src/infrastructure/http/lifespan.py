import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from sqlalchemy import select

from src.application.services.monitoring_service import MonitoringService
from src.config import get_settings
from src.domain.inference.entities.model import Model
from src.domain.monitoring.services.alert_manager import (
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from src.infrastructure.http.container import (
    artifact_storage,
    batch_manager,
    drift_calculator,
    ensure_model_exists,
    feature_store,
    find_project_root,
    inference_engine,
    kafka_producer,
    shutdown_event,
)
from src.infrastructure.http.grpc_server import create_grpc_server
from src.infrastructure.persistence.database import Base, engine, get_db
from src.infrastructure.persistence.models import ModelORM
from src.infrastructure.persistence.postgres_drift_repo import (
    PostgresDriftReportRepository,
)
from src.infrastructure.persistence.postgres_log_repo import (
    PostgresPredictionLogRepository,
)
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)

logger = logging.getLogger(__name__)

# --- Self-Healing: Alert rules for Prometheus metrics ---
alert_manager = AlertManager()
alert_manager.register_rule(
    AlertRule(
        name="high_drift_score",
        metric="drift_score",
        threshold=0.3,
        severity=AlertSeverity.CRITICAL,
        comparison="gt",
        cooldown_seconds=300.0,
        description="Data drift score exceeds critical threshold",
    )
)
alert_manager.register_rule(
    AlertRule(
        name="moderate_drift_score",
        metric="drift_score",
        threshold=0.1,
        severity=AlertSeverity.WARNING,
        comparison="gt",
        cooldown_seconds=120.0,
        description="Data drift score exceeds warning threshold",
    )
)

# Monitoring interval (seconds) — Production: 30s to avoid false-positive flood
MONITORING_INTERVAL_SECONDS = 30


def _load_reference_data() -> list[float]:
    """Load real reference data from the training pipeline output."""
    root = find_project_root()
    ref_paths = [
        root / "data" / "reference_data.json",
        root / "models" / "data" / "reference_data.json",
    ]
    for ref_path in ref_paths:
        if ref_path.exists():
            with open(ref_path) as f:
                data = json.load(f)
            distributions = data.get("reference_distributions", {})
            first_key = next(iter(distributions), None)
            if first_key:
                logger.info(
                    "✅ Loaded %d real reference points from %s",
                    len(distributions[first_key]),
                    ref_path,
                )
                return cast(list[float], distributions[first_key])

    logger.warning("⚠️ No reference data found, using empty reference")
    return []


def _load_real_features() -> list[dict[str, Any]]:
    """Load real feature records seeded from the German Credit dataset."""
    root = find_project_root()
    feat_path = root / "data" / "reference_features.json"
    if feat_path.exists():
        with open(feat_path) as f:
            records = json.load(f)
        logger.info("✅ Loaded %d real feature records from %s", len(records), feat_path)
        return cast(list[dict[str, Any]], records)
    return []


def _load_real_metrics() -> dict[str, Any]:
    """Load real training metrics from the model training output."""
    _settings = get_settings()
    fs_model_id = _settings.DEFAULT_MODEL_ID.replace("-", "_")
    root = find_project_root()
    metrics_path = root / "models" / fs_model_id / _settings.DEFAULT_MODEL_VERSION / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return cast(dict[str, Any], json.load(f))
    return {"accuracy": 0.0, "f1_score": 0.0}


async def run_monitoring_loop() -> None:
    """Background task that periodically checks for data drift."""
    logger.info("🚀 Starting Drift Monitoring Loop...")
    reference_data = _load_reference_data()

    if not reference_data:
        logger.warning("⚠️ No reference data — drift monitoring disabled")
        return

    while not shutdown_event.is_set():
        try:
            # Production monitoring interval
            for _ in range(MONITORING_INTERVAL_SECONDS * 10):
                if shutdown_event.is_set():
                    return
                await asyncio.sleep(0.1)

            async for db in get_db():
                log_repo = PostgresPredictionLogRepository(db)
                drift_repo = PostgresDriftReportRepository(db)
                ms = MonitoringService(
                    log_repo,
                    drift_calculator,
                    drift_repo,
                    alert_manager=alert_manager,
                )

                await ms.check_drift(
                    model_id=get_settings().DEFAULT_MODEL_ID,
                    reference_data=reference_data,
                    feature_index=0,
                )
                break
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("⚠️ Monitoring Loop Error: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    grpc_server = None

    async for db in get_db():
        model_repo = PostgresModelRegistry(db)
        
        grpc_server = create_grpc_server(
            model_repo=model_repo,
            inference_engine=inference_engine,
            batch_manager=batch_manager,
            feature_store=feature_store,
            artifact_storage=artifact_storage,
            port=50051,
        )

        _settings = get_settings()
        _model_id = _settings.DEFAULT_MODEL_ID
        _model_version = _settings.DEFAULT_MODEL_VERSION

        result = await db.execute(
            select(ModelORM).where(
                ModelORM.id == _model_id, ModelORM.version == _model_version
            )
        )
        if not result.scalar_one_or_none():
            try:
                real_model_path = ensure_model_exists(_model_id, _model_version)
                real_metrics = _load_real_metrics()
                logger.info(
                    "✅ Seeding model %s:%s from %s",
                    _model_id, _model_version, real_model_path,
                )
                seed_model = Model(
                    id=_model_id,
                    version=_model_version,
                    uri=f"local://{real_model_path}",
                    framework="onnx",
                    metadata={
                        "features": [
                            "duration",
                            "credit_amount",
                            "installment_commitment",
                            "residence_since",
                            "age",
                            "existing_credits",
                            "num_dependents",
                            "checking_status",
                            "credit_history",
                            "purpose",
                            "savings_status",
                            "employment",
                            "personal_status",
                            "other_parties",
                            "property_magnitude",
                            "other_payment_plans",
                            "housing",
                            "job",
                            "own_telephone",
                            "foreign_worker",
                            "credit_per_month",
                            "age_credit_ratio",
                            "installment_credit_ratio",
                            "age_employment_score",
                            "credit_risk_density",
                            "duration_installment",
                            "checking_savings_interact",
                            "age_checking_interact",
                            "credit_existing_interact",
                            "log_credit_amount",
                        ],
                        "role": "champion",
                        "metrics": real_metrics,
                        "dataset": "german-credit-openml",
                    },
                )
                await model_repo.save(seed_model)
                await model_repo.update_stage(_model_id, _model_version, "champion")
                await db.commit()
                logger.info("✅ Registered model with real metrics: %s", real_metrics)
            except Exception as e:
                logger.error("❌ Failed to seed model: %s", e)

        real_features = _load_real_features()
        if real_features:
            for record in real_features:
                await feature_store.add_features(record["entity_id"], record["features"])
            logger.info("✅ Seeded %d real feature records", len(real_features))

        await feature_store.add_features(
            "customer-good",
            {
                    "duration": 0.5,
                    "credit_amount": -0.3,
                    "installment_commitment": 0.8,
                    "residence_since": 0.5,
                    "age": 0.5,
                    "existing_credits": 0.5,
                    "num_dependents": 0.5,
                    "checking_status": 0.5,
                    "credit_history": 0.5,
                    "purpose": 0.5,
                    "savings_status": 0.5,
                    "employment": 0.5,
                    "personal_status": 0.5,
                    "other_parties": 0.5,
                    "property_magnitude": 0.5,
                    "other_payment_plans": 0.5,
                    "housing": 0.5,
                    "job": 0.5,
                    "own_telephone": 0.5,
                    "foreign_worker": 0.5,
                    "credit_per_month": 0.5,
                    "age_credit_ratio": 0.5,
                    "installment_credit_ratio": 0.5,
                    "age_employment_score": 0.5,
                    "credit_risk_density": 0.5,
                    "duration_installment": 0.5,
                    "checking_savings_interact": 0.5,
                    "age_checking_interact": 0.5,
                    "credit_existing_interact": 0.5,
                    "log_credit_amount": 0.5,
                },
            )
        break

    shutdown_event.clear()
    
    if grpc_server:
        await grpc_server.start()
        
    await kafka_producer.start()
    monitor_task = asyncio.create_task(run_monitoring_loop())

    yield

    logger.info("🧹 Lifespan shutdown started...")
    shutdown_event.set()
    monitor_task.cancel()
    
    if grpc_server:
        await grpc_server.stop(grace=2.0)
        
    await batch_manager.stop()
    await kafka_producer.stop()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except (TimeoutError, asyncio.CancelledError):
        pass
    await engine.dispose()
    logger.info("✅ Cleanup complete.")
