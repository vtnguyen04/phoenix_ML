import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from sqlalchemy import select

from src.application.handlers.retrain_handler import RetrainHandler
from src.application.services.monitoring_service import MonitoringService
from src.domain.inference.entities.model import Model
from src.infrastructure.http.container import (
    batch_manager,
    drift_calculator,
    ensure_model_exists,
    feature_store,
    find_project_root,
    kafka_producer,
    model_evaluator,
    shutdown_event,
)
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


async def run_monitoring_loop() -> None:
    """Background task that periodically checks for data drift."""
    logger.info("🚀 Starting Drift Monitoring Loop...")
    reference_data = np.random.normal(0, 1, 100).tolist()

    while not shutdown_event.is_set():
        try:
            for _ in range(50):
                if shutdown_event.is_set():
                    return
                await asyncio.sleep(0.1)

            async for db in get_db():
                log_repo = PostgresPredictionLogRepository(db)
                drift_repo = PostgresDriftReportRepository(db)
                model_repo = PostgresModelRegistry(db)
                rh = RetrainHandler(find_project_root(), model_repo, model_evaluator)
                ms = MonitoringService(log_repo, drift_calculator, drift_repo, rh)

                await ms.check_drift(
                    model_id="credit-risk",
                    reference_data=reference_data,
                    feature_index=0,
                )
                break
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("⚠️ Monitoring Loop Error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async for db in get_db():
        model_repo = PostgresModelRegistry(db)

        result = await db.execute(
            select(ModelORM).where(ModelORM.id == "credit-risk", ModelORM.version == "v1")
        )
        if not result.scalar_one_or_none():
            try:
                real_model_path = ensure_model_exists()
                logger.info("✅ Seeding model from %s", real_model_path)
                credit_model_v1 = Model(
                    id="credit-risk",
                    version="v1",
                    uri=f"local://{real_model_path}",
                    framework="onnx",
                    metadata={
                        "features": ["income", "debt", "age", "credit_history"],
                        "role": "champion",
                        "metrics": {"accuracy": 0.85, "f1_score": 0.84},
                    },
                )
                await model_repo.save(credit_model_v1)
                await model_repo.update_stage("credit-risk", "v1", "champion")
                await db.commit()
                logger.info("✅ Successfully registered Credit Risk model v1")
            except Exception as e:
                logger.error("❌ Failed to seed model: %s", e)

        await feature_store.add_features(
            "customer-good",
            {"income": 2.0, "debt": -1.5, "age": 1.0, "credit_history": 1.5},
        )
        break

    shutdown_event.clear()
    await kafka_producer.start()
    monitor_task = asyncio.create_task(run_monitoring_loop())

    yield

    logger.info("🧹 Lifespan shutdown started...")
    shutdown_event.set()
    monitor_task.cancel()
    await batch_manager.stop()
    await kafka_producer.stop()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except (TimeoutError, asyncio.CancelledError):
        pass
    await engine.dispose()
    logger.info("✅ Cleanup complete.")
