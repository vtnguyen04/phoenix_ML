"""Tests for environment configuration profiles."""

import pytest

from src.config.environment import (
    Environment,
    get_environment_profile,
)


class TestEnvironmentProfiles:
    def test_development_profile(self) -> None:
        p = get_environment_profile("development")
        assert p.name == Environment.DEVELOPMENT
        assert p.debug is True
        assert p.log_level == "DEBUG"
        assert p.auth_enabled is False
        assert p.workers == 1
        assert p.reload is True

    def test_staging_profile(self) -> None:
        p = get_environment_profile("staging")
        assert p.name == Environment.STAGING
        assert p.debug is False
        assert p.log_level == "INFO"
        assert p.auth_enabled is True
        assert p.workers == 2

    def test_production_profile(self) -> None:
        p = get_environment_profile("production")
        assert p.name == Environment.PRODUCTION
        assert p.debug is False
        assert p.log_level == "WARNING"
        assert p.auth_enabled is True
        assert p.rate_limit_enabled is True
        assert p.workers == 4

    def test_unknown_falls_back_to_dev(self) -> None:
        p = get_environment_profile("unknown")
        assert p.name == Environment.DEVELOPMENT

    def test_case_insensitive(self) -> None:
        p = get_environment_profile("PRODUCTION")
        assert p.name == Environment.PRODUCTION

    def test_profiles_are_frozen(self) -> None:
        p = get_environment_profile("production")
        with pytest.raises(AttributeError):
            p.debug = True  # type: ignore[misc]
