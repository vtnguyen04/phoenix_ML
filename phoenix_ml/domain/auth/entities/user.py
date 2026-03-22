"""User entity and roles for authentication."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime


class UserRole(str, enum.Enum):
    """Role-based access control roles."""

    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    API_CONSUMER = "api_consumer"


@dataclass
class User:
    """Domain entity representing an authenticated user."""

    id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole = UserRole.API_CONSUMER
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def has_role(self, required: UserRole) -> bool:
        """Check if user has at least the required role level."""
        hierarchy: dict[UserRole, int] = {
            UserRole.API_CONSUMER: 0,
            UserRole.DATA_SCIENTIST: 1,
            UserRole.ADMIN: 2,
        }
        return hierarchy.get(self.role, 0) >= hierarchy.get(required, 0)
