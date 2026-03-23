"""Global exception handlers for unified API error responses.

All exceptions are caught and returned as JSON with ``error_code``,
``message``, optional ``details``, and ``correlation_id``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from phoenix_ml.infrastructure.http.middleware.correlation_middleware import (
    correlation_id_var,
)

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error with structured error code."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str = ""):
        detail = f"{resource} '{identifier}'" if identifier else resource
        super().__init__(
            message=f"{detail} not found",
            error_code="NOT_FOUND",
            status_code=404,
        )


class ValidationError(AppError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details or {},
        )


class AuthenticationError(AppError):
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
        )


class AuthorizationError(AppError):
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
        )


class RateLimitError(AppError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
        )


def _build_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build standardized error response."""
    corr_id = correlation_id_var.get("")
    body: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
    }
    if details:
        body["details"] = details
    if corr_id:
        body["correlation_id"] = corr_id
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(AppError)
    async def app_error_handler(
        _request: Request, exc: AppError
    ) -> JSONResponse:
        logger.warning(
            "AppError [%s]: %s", exc.error_code, exc, exc_info=False
        )
        return _build_error_response(
            exc.status_code, exc.error_code, str(exc), exc.details
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMIT_EXCEEDED",
            500: "INTERNAL_ERROR",
        }
        error_code = code_map.get(exc.status_code, "HTTP_ERROR")
        return _build_error_response(
            exc.status_code, error_code, str(exc.detail)
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return _build_error_response(
            500, "INTERNAL_ERROR", "An unexpected error occurred"
        )
