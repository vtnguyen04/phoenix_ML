"""Correlation-ID middleware — assigns unique ID per request for tracing."""

from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Context variable accessible from any code path during the request.
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

logger = logging.getLogger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Assign or forward X-Correlation-ID, and set it in contextvars."""

    HEADER = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Any:
        # Accept from client, or generate new
        cid = request.headers.get(self.HEADER, str(uuid.uuid4()))
        correlation_id_var.set(cid)

        response: Response = await call_next(request)
        response.headers[self.HEADER] = cid
        return response
