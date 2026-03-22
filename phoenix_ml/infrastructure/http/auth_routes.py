"""Auth routes — login, register, refresh, user info."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from phoenix_ml.domain.auth.entities.user import User, UserRole
from phoenix_ml.infrastructure.http.middleware.jwt_middleware import (
    get_auth_service,
    get_current_user,
    require_role,
)

logger = logging.getLogger(__name__)
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Request / Response schemas ────────────────────────────────────


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=8)
    role: str = "api_consumer"


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_active: bool


# ── Endpoints ─────────────────────────────────────────────────────


@auth_router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    """Authenticate and return JWT tokens."""
    auth_service = get_auth_service()
    user = auth_service.authenticate(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return TokenResponse(
        access_token=auth_service.create_access_token(user),
        refresh_token=auth_service.create_refresh_token(user),
    )


@auth_router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    body: RegisterRequest,
    _: User = Depends(require_role(UserRole.ADMIN)),  # noqa: B008
) -> UserResponse:
    """Register a new user (admin only)."""
    auth_service = get_auth_service()
    try:
        role = UserRole(body.role)
    except ValueError:
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Valid: {[r.value for r in UserRole]}",
        )
    try:
        user = auth_service.register(body.username, body.email, body.password, role)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))  # noqa: B904
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest) -> TokenResponse:
    """Refresh access token using a valid refresh token."""
    auth_service = get_auth_service()
    payload = auth_service.decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    username = payload.get("sub")
    user = auth_service.get_user(username) if username else None
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return TokenResponse(
        access_token=auth_service.create_access_token(user),
        refresh_token=auth_service.create_refresh_token(user),
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(
    user: User = Depends(get_current_user),  # noqa: B008
) -> UserResponse:
    """Return current authenticated user info."""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
    )
