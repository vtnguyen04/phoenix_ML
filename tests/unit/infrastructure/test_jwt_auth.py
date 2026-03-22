"""Tests for JWT authentication middleware and AuthService."""

import pytest

from phoenix_ml.domain.auth.entities.user import User, UserRole
from phoenix_ml.domain.auth.services.auth_service import AuthService


@pytest.fixture
def auth_service() -> AuthService:
    return AuthService(
        secret_key="test-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_minutes=60,
    )


class TestUserRole:
    def test_role_hierarchy_admin_has_all(self) -> None:
        user = User(
            id="1", username="a", email="a@a.com", hashed_password="x", role=UserRole.ADMIN
        )
        assert user.has_role(UserRole.API_CONSUMER) is True
        assert user.has_role(UserRole.DATA_SCIENTIST) is True
        assert user.has_role(UserRole.ADMIN) is True

    def test_role_hierarchy_consumer_limited(self) -> None:
        user = User(
            id="2", username="b", email="b@b.com", hashed_password="x", role=UserRole.API_CONSUMER
        )
        assert user.has_role(UserRole.API_CONSUMER) is True
        assert user.has_role(UserRole.DATA_SCIENTIST) is False
        assert user.has_role(UserRole.ADMIN) is False

    def test_role_hierarchy_data_scientist_mid(self) -> None:
        user = User(
            id="3",
            username="c",
            email="c@c.com",
            hashed_password="x",
            role=UserRole.DATA_SCIENTIST,
        )
        assert user.has_role(UserRole.API_CONSUMER) is True
        assert user.has_role(UserRole.DATA_SCIENTIST) is True
        assert user.has_role(UserRole.ADMIN) is False


class TestPasswordHashing:
    def test_hash_and_verify(self, auth_service: AuthService) -> None:
        hashed = auth_service.hash_password("my_secret")
        assert hashed != "my_secret"
        assert auth_service.verify_password("my_secret", hashed) is True

    def test_wrong_password_fails(self, auth_service: AuthService) -> None:
        hashed = auth_service.hash_password("correct")
        assert auth_service.verify_password("wrong", hashed) is False


class TestTokenCreation:
    def test_create_and_decode_access_token(self, auth_service: AuthService) -> None:
        user = User(
            id="1", username="testuser", email="t@t.com", hashed_password="x", role=UserRole.ADMIN
        )
        token = auth_service.create_access_token(user)
        assert isinstance(token, str)

        payload = auth_service.decode_token(token)
        assert payload is not None
        assert payload["sub"] == "testuser"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"

    def test_create_refresh_token(self, auth_service: AuthService) -> None:
        user = User(id="2", username="user2", email="u@u.com", hashed_password="x")
        token = auth_service.create_refresh_token(user)
        payload = auth_service.decode_token(token)
        assert payload is not None
        assert payload["type"] == "refresh"

    def test_invalid_token_returns_none(self, auth_service: AuthService) -> None:
        assert auth_service.decode_token("invalid.token.here") is None

    def test_wrong_secret_returns_none(self, auth_service: AuthService) -> None:
        user = User(id="3", username="u3", email="u3@u.com", hashed_password="x")
        token = auth_service.create_access_token(user)

        other_service = AuthService(secret_key="other-secret")
        assert other_service.decode_token(token) is None


class TestUserManagement:
    def test_bootstrap_admin_exists(self, auth_service: AuthService) -> None:
        admin = auth_service.get_user("admin")
        assert admin is not None
        assert admin.role == UserRole.ADMIN

    def test_authenticate_admin(self, auth_service: AuthService) -> None:
        user = auth_service.authenticate("admin", "admin")
        assert user is not None
        assert user.username == "admin"

    def test_authenticate_wrong_password(self, auth_service: AuthService) -> None:
        assert auth_service.authenticate("admin", "wrongpass") is None

    def test_authenticate_unknown_user(self, auth_service: AuthService) -> None:
        assert auth_service.authenticate("unknown", "pass") is None

    def test_register_new_user(self, auth_service: AuthService) -> None:
        user = auth_service.register("newuser", "new@new.com", "password123")
        assert user.username == "newuser"
        assert user.role == UserRole.API_CONSUMER

        # Can authenticate
        found = auth_service.authenticate("newuser", "password123")
        assert found is not None

    def test_register_duplicate_raises(self, auth_service: AuthService) -> None:
        auth_service.register("dup", "dup@d.com", "pass1234")
        with pytest.raises(ValueError, match="already exists"):
            auth_service.register("dup", "dup2@d.com", "pass5678")

    def test_register_with_role(self, auth_service: AuthService) -> None:
        user = auth_service.register(
            "scientist", "sci@s.com", "pass1234", role=UserRole.DATA_SCIENTIST
        )
        assert user.role == UserRole.DATA_SCIENTIST
