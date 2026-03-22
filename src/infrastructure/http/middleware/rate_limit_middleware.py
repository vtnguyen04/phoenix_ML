"""Redis-backed sliding window rate limiter middleware."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config.auth import AuthSettings

logger = logging.getLogger(__name__)


class _InMemoryRateLimiter:
    """Simple in-memory sliding window counter. Production → use Redis."""

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        """Check if a request is allowed.

        Returns (allowed, remaining).
        """
        now = time.monotonic()
        cutoff = now - window_seconds

        # Clean old entries
        if key not in self._requests:
            self._requests[key] = []
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        count = len(self._requests[key])
        if count >= limit:
            return False, 0

        self._requests[key].append(now)
        return True, limit - count - 1


_limiter = _InMemoryRateLimiter()
_settings = AuthSettings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Any:
        if not _settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        # Skip health checks and metrics
        path = request.url.path
        if path in ("/health", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Determine limit based on endpoint
        limit = _settings.RATE_LIMIT_PER_MINUTE
        if "/predict" in path:
            limit = _settings.RATE_LIMIT_PREDICT_PER_MINUTE

        # Key = IP + path prefix
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get(_settings.API_KEY_HEADER, "")
        raw_key = f"{client_ip}:{api_key}" if api_key else client_ip
        rate_key = hashlib.md5(raw_key.encode()).hexdigest()  # noqa: S324

        allowed, remaining = _limiter.is_allowed(rate_key, limit)

        if not allowed:
            logger.warning("Rate limit exceeded for %s on %s", client_ip, path)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": "60"},
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
