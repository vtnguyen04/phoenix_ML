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
from src.infrastructure.bootstrap.container import (
    artifact_storage,
    batch_manager,
    drift_calculator,
    ensure_model_exists,
    event_bus,
    feature_store,
    find_project_root,
    inference_engine,
    kafka_producer,
    plugin_registry,
    shutdown_event,
)
from src.infrastructure.grpc.grpc_server import create_grpc_server
from src.infrastructure.bootstrap.model_config_loader import (
    load_all_model_configs,
    load_features_from_metrics,
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

settings = get_settings()

# Monitoring interval (seconds) — Production: 30s to avoid false-positive flood
MONITORING_INTERVAL_SECONDS = settings.MONITORING_INTERVAL_SECONDS


def _load_reference_data_for_model(model_id: str) -> list[float]:
    """Load per-model reference data for drift detection.

    Resolution order:
    1. models/<model_id>/v1/reference_data.json
    2. Global data/reference_data.json (fallback)
    """
    root = find_project_root()
    fs_id = model_id.replace("-", "_")
    ref_paths = [
        root / "models" / fs_id / "v1" / "reference_data.json",
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
                    "✅ Loaded %d reference points for %s from %s",
                    len(distributions[first_key]),
                    model_id,
                    ref_path,
                )
                return cast(list[float], distributions[first_key])

    logger.warning("⚠️ No reference data for %s — drift monitoring disabled", model_id)
    return []


def _load_real_features() -> list[dict[str, Any]]:
    """Load real feature records from the seeded reference data."""
    root = find_project_root()
    feat_path = root / "data" / "reference_features.json"
    if feat_path.exists():
        with open(feat_path) as f:
            records = json.load(f)
        logger.info("✅ Loaded %d real feature records from %s", len(records), feat_path)
        return cast(list[dict[str, Any]], records)
    return []


def _load_real_metrics(model_id: str, version: str) -> dict[str, Any]:
    """Load real training metrics from the model training output.

    Dynamically resolves the metrics path from model_id and version,
    making this function model-agnostic.
    """
    fs_model_id = model_id.replace("-", "_")
    root = find_project_root()
    metrics_path = root / "models" / fs_model_id / version / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return cast(dict[str, Any], json.load(f))
    return {"accuracy": 0.0, "f1_score": 0.0}


def _resolve_feature_names(model_id: str, version: str) -> list[str]:
    """Resolve feature names dynamically from model config or metrics.json.

    Resolution order:
    1. Model config YAML (model_configs/<model_id>.yaml -> feature_names)
    2. Training metrics (models/<model_id>/<version>/metrics.json -> all_features)
    3. Empty list (model starts without named features)
    """
    _settings = get_settings()
    root = find_project_root()

    # 1. Try model config YAML
    config_dir = root / _settings.MODEL_CONFIG_DIR
    configs = load_all_model_configs(config_dir)
    if model_id in configs and configs[model_id].has_named_features:
        features = list(configs[model_id].feature_names)
        logger.info("📋 Loaded %d features from config: %s", len(features), model_id)
        return features

    # 2. Try metrics.json
    fs_model_id = model_id.replace("-", "_")
    metrics_path = root / "models" / fs_model_id / version / "metrics.json"
    features = load_features_from_metrics(metrics_path)
    if features:
        logger.info("📊 Loaded %d features from metrics.json", len(features))
        return features

    # 3. No features found
    logger.info(
        "ℹ️ No feature names found for %s:%s — starting without named features",
        model_id, version,
    )
    return []

async def run_monitoring_loop() -> None:
    """Background task that periodically checks for data drift on ALL models.

    Each model uses its own:
    - Reference data (per-model reference_data.json)
    - Drift test type (from model_configs/*.yaml → monitoring.drift_test)
    - Feature index (defaults to 0 for tabular, could be extended)
    """
    logger.info("🚀 Starting Multi-Model Drift Monitoring Loop...")

    # Build per-model monitoring config from YAML
    root = find_project_root()
    _settings = get_settings()
    config_dir = root / _settings.MODEL_CONFIG_DIR
    model_configs = load_all_model_configs(config_dir)

    if not model_configs:
        logger.warning("⚠️ No model configs found — drift monitoring disabled")
        return

    # Load per-model reference data
    model_refs: dict[str, list[float]] = {}
    for mid in model_configs:
        ref = _load_reference_data_for_model(mid)
        if ref:
            model_refs[mid] = ref

    if not model_refs:
        logger.warning("⚠️ No reference data for any model — drift monitoring disabled")
        return

    logger.info(
        "📋 Monitoring %d models: %s",
        len(model_refs),
        list(model_refs.keys()),
    )

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
                    event_bus=event_bus,
                )

                # Check drift for EVERY configured model
                for mid, ref_data in model_refs.items():
                    cfg = model_configs.get(mid)
                    test_type = cfg.monitoring_drift_test if cfg else "ks"
                    try:
                        await ms.check_drift(
                            model_id=mid,
                            reference_data=ref_data,
                            feature_index=0,
                            test_type=test_type,
                        )
                    except ValueError:
                        # Not enough prediction data yet — skip silently
                        pass
                    except Exception as e:
                        logger.warning(
                            "⚠️ Drift check failed for %s: %s", mid, e
                        )
                break
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("⚠️ Monitoring Loop Error: %s", e, exc_info=True)
            # Sleep before retrying to prevent tight error loops
            await asyncio.sleep(MONITORING_INTERVAL_SECONDS)


async def _seed_model_if_needed(
    db: Any,
    model_repo: PostgresModelRegistry,
    model_id: str,
    model_version: str,
) -> None:
    """Seed model into registry if not already present."""
    result = await db.execute(
        select(ModelORM).where(
            ModelORM.id == model_id, ModelORM.version == model_version
        )
    )
    if result.scalar_one_or_none():
        return

    try:
        real_model_path = ensure_model_exists(model_id, model_version)
        real_metrics = _load_real_metrics(model_id, model_version)
        feature_names = _resolve_feature_names(model_id, model_version)
        logger.info(
            "✅ Seeding model %s:%s from %s",
            model_id, model_version, real_model_path,
        )

        # Build metadata dynamically — no hardcoded features/datasets
        model_metadata: dict[str, Any] = {
            "role": "champion",
            "metrics": real_metrics,
        }
        if feature_names:
            model_metadata["features"] = feature_names
        if real_metrics.get("dataset"):
            model_metadata["dataset"] = real_metrics["dataset"]

        seed_model = Model(
            id=model_id,
            version=model_version,
            uri=f"local://{real_model_path}",
            framework="onnx",
            metadata=model_metadata,
        )
        await model_repo.save(seed_model)
        await model_repo.update_stage(model_id, model_version, "champion")
        await db.commit()
        logger.info("✅ Registered model with real metrics: %s", real_metrics)

        # Publish metrics via event bus (Observer Pattern — no direct Prometheus dep)
        from src.domain.shared.domain_events import ModelRetrained  # noqa: PLC0415

        event_bus.publish(ModelRetrained(
            model_id=model_id,
            version=model_version,
            metrics=real_metrics,
            promoted=True,
        ))
    except Exception as e:
        logger.error("❌ Failed to seed model: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    grpc_server = None

    async for db in get_db():
        model_repo = PostgresModelRegistry(db)

        try:
            grpc_server = create_grpc_server(
                model_repo=model_repo,
                inference_engine=inference_engine,
                batch_manager=batch_manager,
                feature_store=feature_store,
                artifact_storage=artifact_storage,
                port=50051,
            )
        except RuntimeError as e:
            logger.warning("⚠️ gRPC server skipped (port in use): %s", e)
            grpc_server = None

        _settings = get_settings()
        _model_id = _settings.DEFAULT_MODEL_ID
        _model_version = _settings.DEFAULT_MODEL_VERSION

        # Seed default model (if configured)
        if _model_id:
            await _seed_model_if_needed(db, model_repo, _model_id, _model_version)

        # Seed ALL models from model_configs/ directory
        root = find_project_root()
        config_dir = root / _settings.MODEL_CONFIG_DIR
        model_configs = load_all_model_configs(config_dir)
        for cfg_id, cfg in model_configs.items():
            if cfg_id != _model_id:
                cfg_version = cfg.version or _model_version
                await _seed_model_if_needed(
                    db, model_repo, cfg_id, cfg_version,
                )

        # Seed feature store from reference data (model-agnostic)
        real_features = _load_real_features()
        if real_features:
            for record in real_features:
                await feature_store.add_features(record["entity_id"], record["features"])
            logger.info("✅ Seeded %d real feature records", len(real_features))

        break

    # Log model configs and plugin registry state
    if model_configs:
        logger.info(
            "📋 Loaded %d model configs: %s",
            len(model_configs),
            list(model_configs.keys()),
        )
    if plugin_registry.registered_models:
        logger.info("🔌 Plugins registered: %s", plugin_registry.summary())

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
