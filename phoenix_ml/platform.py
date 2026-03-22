"""PhoenixPlatform — Programmatic entry point for Phoenix ML framework.

Usage::

    from phoenix_ml import PhoenixPlatform

    platform = PhoenixPlatform(
        database_url="postgresql+asyncpg://user:pass@cloud-db:5432/mydb",
        redis_url="redis://cloud-redis:6379",
        mlflow_uri="http://cloud-mlflow:5000",
        model_configs_dir="./my_models/",
    )
    platform.serve(host="0.0.0.0", port=8000)

All parameters are optional — sensible defaults used when not provided:
  - database_url: SQLite (no database server needed)
  - redis_url:    In-memory cache (no Redis needed)
  - mlflow_uri:   No MLflow logging
  - kafka_url:    Events logged locally (no Kafka needed)
"""

import logging
import os
from pathlib import Path

import uvicorn

logger = logging.getLogger(__name__)


class PhoenixPlatform:
    """Main framework entry point with programmatic configuration.

    Attributes set here override environment variables and defaults.
    Any parameter set to None falls back to env var → default.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # ── Infrastructure URLs ──
        database_url: str | None = None,
        redis_url: str | None = None,
        kafka_url: str | None = None,
        mlflow_uri: str | None = None,
        jaeger_endpoint: str | None = None,
        # ── Framework config ──
        model_configs_dir: str | Path = "model_configs",
        inference_engine: str = "onnx",
        default_model_id: str = "",
        # ── Feature flags ──
        use_redis: bool | None = None,
        debug: bool = False,
    ) -> None:
        # Set env vars BEFORE importing settings (pydantic reads from env)
        if database_url is not None:
            os.environ["DATABASE_URL"] = database_url
        if redis_url is not None:
            os.environ["REDIS_URL"] = redis_url
            if use_redis is None:
                os.environ["USE_REDIS"] = "true"
        if use_redis is not None:
            os.environ["USE_REDIS"] = str(use_redis).lower()
        if kafka_url is not None:
            os.environ["KAFKA_URL"] = kafka_url
        if mlflow_uri is not None:
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        if jaeger_endpoint is not None:
            os.environ["JAEGER_ENDPOINT"] = jaeger_endpoint
        if debug:
            os.environ["DEBUG"] = "true"

        os.environ["INFERENCE_ENGINE"] = inference_engine
        os.environ["MODEL_CONFIG_DIR"] = str(model_configs_dir)
        if default_model_id:
            os.environ["DEFAULT_MODEL_ID"] = default_model_id

        self._model_configs_dir = Path(model_configs_dir)
        self._debug = debug

        logger.info("PhoenixPlatform initialized")
        logger.info("  database:   %s", os.environ.get("DATABASE_URL", "(default: SQLite)"))
        redis_status = (
            "enabled" if os.environ.get("USE_REDIS") == "true"
            else "disabled (in-memory)"
        )
        logger.info("  redis:      %s", redis_status)
        logger.info("  mlflow:     %s", os.environ.get("MLFLOW_TRACKING_URI", "(disabled)"))
        logger.info("  configs:    %s", self._model_configs_dir)
        logger.info("  engine:     %s", inference_engine)

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        reload: bool = False,
    ) -> None:
        """Start the Phoenix ML API server.

        Args:
            host: Bind address.
            port: Port number.
            workers: Number of uvicorn workers.
            reload: Auto-reload on code changes (dev only).
        """
        logger.info("Starting Phoenix ML server on %s:%d", host, port)
        uvicorn.run(
            "phoenix_ml.infrastructure.http.fastapi_server:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="debug" if self._debug else "info",
        )

    @property
    def app(self):  # type: ignore[no-untyped-def]
        """Get the FastAPI app instance for ASGI deployment (Gunicorn, etc.)."""
        from phoenix_ml.infrastructure.http.fastapi_server import app  # noqa: PLC0415

        return app
