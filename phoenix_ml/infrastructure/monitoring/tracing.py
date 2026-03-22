"""
OpenTelemetry Distributed Tracing Setup.
Configures a tracer provider with OTLP/Jaeger exporter for end-to-end
request tracing across the Phoenix ML Platform.
"""

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None


def init_tracing(
    service_name: str = "phoenix-ml-platform",
    otlp_endpoint: str = "http://localhost:4317",
    enable_console: bool = False,
) -> TracerProvider:
    """
    Initialize the OpenTelemetry TracerProvider with OTLP exporter
    (Jaeger-compatible).
    """
    global _tracer_provider  # noqa: PLW0603

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("✅ OTLP tracing exporter configured → %s", otlp_endpoint)
        else:
            logger.info("⏭️ OTLP tracing skipped: endpoint is empty")
    except ImportError:
        logger.info("OTLP exporter not installed, tracing export disabled")
    except Exception as e:
        logger.warning("⚠️ Failed to configure OTLP exporter: %s", e)

    if enable_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    return provider


def get_tracer(name: str = "phoenix-ml") -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Gracefully shutdown the tracer provider."""
    global _tracer_provider  # noqa: PLW0603
    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None
