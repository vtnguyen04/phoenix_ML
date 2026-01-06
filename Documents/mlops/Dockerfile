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

# Create a non-root user for security
RUN groupadd -r phoenix && useradd -r -g phoenix phoenix

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Switch to non-root user
USER phoenix

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "src.infrastructure.http.fastapi_server"]
