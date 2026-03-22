"""AuthService — JWT token creation, validation, and password hashing."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from phoenix_ml.domain.auth.entities.user import User, UserRole


class AuthService:
    """Pure domain service for authentication logic."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_minutes: int = 1440,
    ) -> None:
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_expire = access_token_expire_minutes
        self._refresh_expire = refresh_token_expire_minutes

        # In-memory user store (production → DB)
        self._users: dict[str, User] = {}
        self._bootstrap_admin()

    def _bootstrap_admin(self) -> None:
        """Create default admin user."""
        admin = User(
            id=str(uuid.uuid4()),
            username="admin",
            email="admin@phoenix-ml.local",
            hashed_password=self.hash_password("admin"),
            role=UserRole.ADMIN,
        )
        self._users[admin.username] = admin

    # ── Password ────────────────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256 with random salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"{salt}${hashed}"

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        """Verify password against stored hash."""
        if "$" not in hashed:
            return False
        salt, stored_hash = hashed.split("$", 1)
        computed = hashlib.sha256(f"{salt}{plain}".encode()).hexdigest()
        return secrets.compare_digest(computed, stored_hash)

    # ── Token ───────────────────────────────────────────────────────

    def create_access_token(self, user: User) -> str:
        expire = datetime.now(UTC) + timedelta(minutes=self._access_expire)
        payload: dict[str, Any] = {
            "sub": user.username,
            "role": user.role.value,
            "exp": expire,
            "type": "access",
        }
        return str(jwt.encode(payload, self._secret_key, algorithm=self._algorithm))

    def create_refresh_token(self, user: User) -> str:
        expire = datetime.now(UTC) + timedelta(minutes=self._refresh_expire)
        payload: dict[str, Any] = {
            "sub": user.username,
            "exp": expire,
            "type": "refresh",
        }
        return str(jwt.encode(payload, self._secret_key, algorithm=self._algorithm))

    def decode_token(self, token: str) -> dict[str, Any] | None:
        """Decode and validate a JWT token. Returns None on failure."""
        try:
            payload: dict[str, Any] = jwt.decode(
                token, self._secret_key, algorithms=[self._algorithm]
            )
            return payload
        except JWTError:
            return None

    # ── User CRUD (in-memory, production → repository) ──────────────

    def authenticate(self, username: str, password: str) -> User | None:
        user = self._users.get(username)
        if user and self.verify_password(password, user.hashed_password):
            return user
        return None

    def register(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.API_CONSUMER,
    ) -> User:
        if username in self._users:
            msg = f"User '{username}' already exists"
            raise ValueError(msg)
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=self.hash_password(password),
            role=role,
        )
        self._users[username] = user
        return user

    def get_user(self, username: str) -> User | None:
        return self._users.get(username)
