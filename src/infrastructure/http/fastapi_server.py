import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.config import get_settings
from src.infrastructure.bootstrap.lifespan import lifespan
from src.infrastructure.http.auth_routes import auth_router
from src.infrastructure.http.data_routes import data_router
from src.infrastructure.http.explain_routes import explain_router
from src.infrastructure.http.feature_routes import feature_router
from src.infrastructure.http.middleware.correlation_middleware import CorrelationMiddleware
from src.infrastructure.http.middleware.rate_limit_middleware import RateLimitMiddleware
from src.infrastructure.http.routes import router
from src.infrastructure.logging.logging_config import configure_logging
from src.infrastructure.monitoring.tracing import init_tracing

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Structured logging ────────────────────────────────────────────
configure_logging(level="DEBUG" if settings.DEBUG else "INFO", json_format=not settings.DEBUG)

# ── Tracing ───────────────────────────────────────────────────────
init_tracing(service_name=settings.APP_NAME, otlp_endpoint=settings.JAEGER_ENDPOINT)

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _instrument_fastapi = True
except ImportError:
    _instrument_fastapi = False
    logger.info("OTEL FastAPI instrumentation not installed, skipping")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

if _instrument_fastapi:
    FastAPIInstrumentor.instrument_app(app)

# ── Middleware Stack (order matters: first added = outermost) ─────
# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Correlation ID (adds X-Correlation-ID to every request/response)
app.add_middleware(CorrelationMiddleware)

# 3. Rate Limiting (blocks excessive requests)
app.add_middleware(RateLimitMiddleware)

# ── Metrics ───────────────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routes (API v1) ──────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1")
app.include_router(feature_router, prefix="/api/v1")
app.include_router(explain_router, prefix="/api/v1")
app.include_router(data_router, prefix="/api/v1")

# ── Backward-compatible routes (no prefix) ────────────────────────
app.include_router(router)
app.include_router(feature_router)
app.include_router(data_router)


def run() -> None:
    """CLI entry point: `phoenix-serve` command."""
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "src.infrastructure.http.fastapi_server:app",
        host=settings.HOST if hasattr(settings, "HOST") else "0.0.0.0",
        port=settings.PORT if hasattr(settings, "PORT") else 8000,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    run()
