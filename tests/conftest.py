"""Root conftest.py — ensures tests use local-friendly settings.

When running pytest locally (outside Docker), we must override Docker-internal
env vars (redis://redis:6379, kafka:9092) that resolve to container DNS names.
This conftest runs before any test module import, preventing Redis/Kafka
connection errors and async event loop conflicts.
"""

import os

# Force override Docker env vars BEFORE any src module imports.
# container.py (module-level) reads Settings() which uses .env file.
# We must override BEFORE that import chain starts.
os.environ["USE_REDIS"] = "false"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./phoenix_test.db"
os.environ["KAFKA_URL"] = ""
os.environ["JAEGER_ENDPOINT"] = ""
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlflow_test"
os.environ.setdefault("DEFAULT_MODEL_ID", "credit-risk")

# Clear the lru_cache on get_settings so it re-reads env vars
from collections.abc import Generator
from unittest.mock import patch

import pytest

from src.config import get_settings  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def disable_grpc_for_tests() -> Generator[None, None, None]:
    """Prevent gRPC from spawning ThreadPoolExecutors that hang pytest teardown."""
    with patch("src.infrastructure.bootstrap.lifespan.create_grpc_server", return_value=None):
        yield


@pytest.fixture(scope="session", autouse=True)
def _shutdown_otel_on_exit() -> Generator[None, None, None]:
    """Shut down OpenTelemetry TracerProvider after all tests.

    fastapi_server.py calls init_tracing() at MODULE LEVEL, which creates
    a TracerProvider with a BatchSpanProcessor daemon thread.
    Without explicit shutdown, that thread blocks process exit.
    """
    yield
    from src.infrastructure.monitoring.tracing import shutdown_tracing

    shutdown_tracing()
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass


def pytest_unconfigure(config: pytest.Config) -> None:
    """Force-terminate the process after pytest finishes all work.

    This is the NUCLEAR FAILSAFE.  By the time pytest_unconfigure runs,
    coverage reports have already been written to XML / stdout.  Any
    remaining non-daemon threads (BatchSpanProcessor, gRPC ThreadPool,
    httpx keep-alive, etc.) would otherwise keep the process alive
    forever on CI.  os._exit(0) bypasses Python's slow interpreter
    shutdown and thread-joining, guaranteeing the CI job completes.
    """
    os._exit(0)
