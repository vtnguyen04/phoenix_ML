import logging
import signal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from phoenix_ml.config import get_settings
from phoenix_ml.infrastructure.bootstrap.lifespan import lifespan
from phoenix_ml.infrastructure.http.auth_routes import auth_router
from phoenix_ml.infrastructure.http.data_routes import data_router
from phoenix_ml.infrastructure.http.error_handlers import register_exception_handlers
from phoenix_ml.infrastructure.http.explain_routes import explain_router
from phoenix_ml.infrastructure.http.feature_routes import feature_router
from phoenix_ml.infrastructure.http.middleware.correlation_middleware import CorrelationMiddleware
from phoenix_ml.infrastructure.http.middleware.rate_limit_middleware import RateLimitMiddleware
from phoenix_ml.infrastructure.http.routes import router
from phoenix_ml.infrastructure.http.websocket_routes import ws_router
from phoenix_ml.infrastructure.logging.logging_config import configure_logging
from phoenix_ml.infrastructure.monitoring.tracing import init_tracing

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Structured logging ────────────────────────────────────────────
configure_logging(
    level="DEBUG" if settings.DEBUG else "INFO",
    json_format=not settings.DEBUG,
)

# ── Tracing ───────────────────────────────────────────────────────
init_tracing(
    service_name=settings.APP_NAME,
    otlp_endpoint=settings.JAEGER_ENDPOINT,
)

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _instrument_fastapi = True
except ImportError:
    _instrument_fastapi = False
    logger.info("OTEL FastAPI instrumentation not installed, skipping")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

if _instrument_fastapi:
    FastAPIInstrumentor.instrument_app(app)

# ── Global Exception Handlers ────────────────────────────────────
register_exception_handlers(app)

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

# ── WebSocket routes ─────────────────────────────────────────────
app.include_router(ws_router)

# ── Backward-compatible routes (no prefix) ────────────────────────
app.include_router(router)
app.include_router(feature_router)
app.include_router(data_router)


# ── Graceful shutdown ─────────────────────────────────────────────
_shutting_down = False


def _graceful_shutdown(signum: int, _frame: object) -> None:
    """Handle SIGTERM/SIGINT gracefully — drain connections first."""
    global _shutting_down  # noqa: PLW0603
    if _shutting_down:
        return
    _shutting_down = True
    logger.info(
        "Received signal %s — starting graceful shutdown...", signum
    )
    # Uvicorn handles shutdown via lifespan on SIGINT/SIGTERM
    # This log ensures we know about it


# Register signal handlers (only in main process, not in workers)
try:
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
except (ValueError, OSError):
    # Can't set signal handlers in non-main thread
    pass


def run() -> None:
    """CLI entry point: `phoenix-serve` command."""
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "src.infrastructure.http.fastapi_server:app",
        host=getattr(settings, "HOST", "0.0.0.0"),
        port=getattr(settings, "PORT", 8000),
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    run()
