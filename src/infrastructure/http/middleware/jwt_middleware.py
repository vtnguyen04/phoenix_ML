"""JWT authentication middleware + API key support."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config.auth import AuthSettings
from src.domain.auth.entities.user import User, UserRole
from src.domain.auth.services.auth_service import AuthService

logger = logging.getLogger(__name__)

_auth_settings = AuthSettings()
_auth_service = AuthService(
    secret_key=_auth_settings.JWT_SECRET_KEY,
    algorithm=_auth_settings.JWT_ALGORITHM,
    access_token_expire_minutes=_auth_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    refresh_token_expire_minutes=_auth_settings.JWT_REFRESH_TOKEN_EXPIRE_MINUTES,
)

_bearer_scheme = HTTPBearer(auto_error=False)


def get_auth_service() -> AuthService:
    """Return the global AuthService singleton."""
    return _auth_service


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> User | None:
    """Extract user from JWT or API key. Returns None if no valid auth."""
    # Try JWT Bearer token first
    if credentials:
        payload = _auth_service.decode_token(credentials.credentials)
        if payload:
            username: str | None = payload.get("sub")
            if username:
                return _auth_service.get_user(username)

    # Try API key
    api_key = request.headers.get(_auth_settings.API_KEY_HEADER)
    if api_key and _auth_settings.API_KEYS:
        valid_keys = [k.strip() for k in _auth_settings.API_KEYS.split(",")]
        if api_key in valid_keys:
            # API key maps to a synthetic api_consumer user
            return User(
                id="api-key-user",
                username="api-key",
                email="",
                hashed_password="",
                role=UserRole.API_CONSUMER,
            )

    return None


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> User:
    """Require authenticated user. Raises 401 if auth is enabled and no valid token."""
    if not _auth_settings.AUTH_ENABLED:
        # Auth disabled → return anonymous admin (development mode)
        return User(
            id="anonymous",
            username="anonymous",
            email="",
            hashed_password="",
            role=UserRole.ADMIN,
        )

    user = await get_current_user_optional(request, credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide valid Bearer token or API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User disabled")
    return user


def require_role(required_role: UserRole) -> Any:
    """Dependency that checks user has at least the required role."""

    async def _check(user: User = Depends(get_current_user)) -> User:  # noqa: B008
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role.value} or higher",
            )
        return user

    return _check
