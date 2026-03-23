"""Programmatic entry point for the Phoenix ML framework.

Wraps configuration, server startup, model loading, and inference
into a single ``PhoenixPlatform`` class for library-mode usage.
"""

import logging
import os
from pathlib import Path

import uvicorn

logger = logging.getLogger(__name__)


class PhoenixPlatform:
    """Convenience facade for configuring and starting the platform.

    Constructor kwargs override env vars; ``None`` falls back to
    env var then default. All infrastructure services are optional.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # ── Database ──
        database_url: str | None = None,
        # ── Cache ──
        redis_url: str | None = None,
        use_redis: bool | None = None,
        # ── Messaging ──
        kafka_url: str | None = None,
        # ── Experiment Tracking ──
        mlflow_uri: str | None = None,
        # ── Pipeline Orchestration ──
        airflow_url: str | None = None,
        airflow_user: str | None = None,
        airflow_password: str | None = None,
        # ── Observability ──
        jaeger_endpoint: str | None = None,
        # ── Framework Config ──
        model_configs_dir: str | Path = "model_configs",
        inference_engine: str = "onnx",
        default_model_id: str = "",
        # ── Server ──
        grpc_port: int | None = None,
        debug: bool = False,
    ) -> None:
        self._apply_env("DATABASE_URL", database_url)
        self._apply_env("REDIS_URL", redis_url)
        self._apply_env("KAFKA_URL", kafka_url)
        self._apply_env("MLFLOW_TRACKING_URI", mlflow_uri)
        self._apply_env("AIRFLOW_API_URL", airflow_url)
        self._apply_env("AIRFLOW_ADMIN_USER", airflow_user)
        self._apply_env("AIRFLOW_ADMIN_PASSWORD", airflow_password)
        self._apply_env("JAEGER_ENDPOINT", jaeger_endpoint)

        # Redis auto-enable when URL is provided
        if redis_url is not None and use_redis is None:
            os.environ["USE_REDIS"] = "true"
        if use_redis is not None:
            os.environ["USE_REDIS"] = str(use_redis).lower()

        if grpc_port is not None:
            os.environ["GRPC_PORT"] = str(grpc_port)
        if debug:
            os.environ["DEBUG"] = "true"

        os.environ["INFERENCE_ENGINE"] = inference_engine
        os.environ["MODEL_CONFIG_DIR"] = str(model_configs_dir)
        if default_model_id:
            os.environ["DEFAULT_MODEL_ID"] = default_model_id

        self._model_configs_dir = Path(model_configs_dir)
        self._debug = debug

        self._log_config(inference_engine)

    @staticmethod
    def _apply_env(key: str, value: str | None) -> None:
        """Set environment variable if value is provided."""
        if value is not None:
            os.environ[key] = value

    def _log_config(self, engine: str) -> None:
        """Log active configuration summary."""
        logger.info("PhoenixPlatform initialized")
        _env = os.environ.get

        services = {
            "database": _env("DATABASE_URL", "SQLite (default)"),
            "redis": (
                _env("REDIS_URL", "")
                if _env("USE_REDIS") == "true"
                else "in-memory (default)"
            ),
            "kafka": _env("KAFKA_URL", "disabled (events local)"),
            "mlflow": _env("MLFLOW_TRACKING_URI", "disabled"),
            "airflow": _env("AIRFLOW_API_URL", "disabled (manual retrain)"),
            "jaeger": _env("JAEGER_ENDPOINT", "disabled"),
            "engine": engine,
            "configs": str(self._model_configs_dir),
        }
        for name, value in services.items():
            logger.info("  %-10s %s", name, value)

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
        """Get the FastAPI app instance (for Gunicorn/ASGI deployment)."""
        from phoenix_ml.infrastructure.http.fastapi_server import (  # noqa: PLC0415
            app,
        )

        return app
