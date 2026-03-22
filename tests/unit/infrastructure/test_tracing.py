from phoenix_ml.infrastructure.monitoring.tracing import (
    get_tracer,
    init_tracing,
    shutdown_tracing,
)


def test_init_tracing_returns_provider() -> None:
    provider = init_tracing(
        service_name="test-service",
        otlp_endpoint="http://localhost:4317",
    )
    assert provider is not None
    shutdown_tracing()


def test_get_tracer_returns_tracer() -> None:
    init_tracing(service_name="test-service")
    tracer = get_tracer("test-module")
    assert tracer is not None
    shutdown_tracing()


def test_shutdown_tracing_is_idempotent() -> None:
    init_tracing(service_name="test-service")
    shutdown_tracing()
    shutdown_tracing()  # Should not raise
