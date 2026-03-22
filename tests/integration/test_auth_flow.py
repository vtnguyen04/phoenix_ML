"""Integration test for auth flow: register → login → access → refresh."""

import pytest

from phoenix_ml.domain.auth.entities.user import UserRole
from phoenix_ml.domain.auth.services.auth_service import AuthService


@pytest.fixture
def auth_service() -> AuthService:
    return AuthService(
        secret_key="integration-test-secret",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_minutes=60,
    )


class TestAuthIntegrationFlow:
    """Full auth lifecycle: register → login → token → access → refresh."""

    def test_full_auth_flow(self, auth_service: AuthService) -> None:
        # 1. Register
        user = auth_service.register(
            username="testuser",
            email="test@test.com",
            password="securepass123",
            role=UserRole.DATA_SCIENTIST,
        )
        assert user.username == "testuser"
        assert user.role == UserRole.DATA_SCIENTIST

        # 2. Login (authenticate)
        auth_user = auth_service.authenticate("testuser", "securepass123")
        assert auth_user is not None
        assert auth_user.id == user.id

        # 3. Create tokens
        access_token = auth_service.create_access_token(auth_user)
        refresh_token = auth_service.create_refresh_token(auth_user)
        assert access_token != refresh_token

        # 4. Decode access token
        payload = auth_service.decode_token(access_token)
        assert payload is not None
        assert payload["sub"] == "testuser"
        assert payload["role"] == "data_scientist"
        assert payload["type"] == "access"

        # 5. Decode refresh token
        refresh_payload = auth_service.decode_token(refresh_token)
        assert refresh_payload is not None
        assert refresh_payload["type"] == "refresh"

        # 6. Create new access token from refresh (simulating /auth/refresh)
        refreshed_user = auth_service.get_user(refresh_payload["sub"])
        assert refreshed_user is not None
        new_access = auth_service.create_access_token(refreshed_user)
        new_payload = auth_service.decode_token(new_access)
        assert new_payload is not None
        assert new_payload["sub"] == "testuser"

    def test_role_based_access(self, auth_service: AuthService) -> None:
        # Register users with different roles
        consumer = auth_service.register("consumer", "c@c.com", "password1234")
        scientist = auth_service.register(
            "scientist", "s@s.com", "password1234", UserRole.DATA_SCIENTIST
        )

        # Consumer can't access data scientist features
        assert consumer.has_role(UserRole.DATA_SCIENTIST) is False
        # Scientist can access consumer features
        assert scientist.has_role(UserRole.API_CONSUMER) is True

    def test_wrong_credentials_rejected(self, auth_service: AuthService) -> None:
        auth_service.register("user1", "u@u.com", "correctpassword")
        assert auth_service.authenticate("user1", "wrongpassword") is None
        assert auth_service.authenticate("nonexistent", "password") is None

    def test_invalid_token_rejected(self, auth_service: AuthService) -> None:
        assert auth_service.decode_token("not.a.valid.token") is None
        assert auth_service.decode_token("") is None
