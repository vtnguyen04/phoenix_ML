# syntax=docker/dockerfile:1

# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: BUILDER
# Use uv for fast dependency installation
# ═══════════════════════════════════════════════════════════════════════
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# --no-dev: Exclude development dependencies (pytest, ruff, mypy)
# --frozen: Require uv.lock to be up-to-date
RUN uv sync --frozen --no-dev --no-install-project

# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: RUNNER
# Minimal runtime image
# ═══════════════════════════════════════════════════════════════════════
FROM python:3.11-slim-bookworm AS runner

# Create a non-root user for security and install curl + gosu
RUN groupadd -r phoenix && useradd -r -g phoenix phoenix && \
    apt-get update && apt-get install -y --no-install-recommends curl gosu && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code, scripts, and data
COPY src /app/src
COPY scripts /app/scripts
COPY data /app/data

# Create writable directories for volumes
RUN mkdir -p /app/models /app/data && chown -R phoenix:phoenix /app/models /app/data

# Copy entrypoint (runs as root to fix volume perms, then drops to phoenix)
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Expose ports (HTTP + gRPC)
EXPOSE 8000 50051

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint fixes permissions then drops to phoenix user
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Start the application
CMD ["uvicorn", "src.infrastructure.http.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
