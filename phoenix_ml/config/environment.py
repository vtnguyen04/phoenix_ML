"""Environment-specific configuration profiles.

Provides dev/staging/production config overrides based on
the ``ENVIRONMENT`` env var.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True)
class EnvironmentProfile:
    """Configuration profile for a specific environment."""

    name: Environment
    debug: bool
    log_level: str
    log_json: bool
    auth_enabled: bool
    rate_limit_enabled: bool
    cors_origins: list[str]
    workers: int
    reload: bool


# Pre-defined profiles
PROFILES: dict[Environment, EnvironmentProfile] = {
    Environment.DEVELOPMENT: EnvironmentProfile(
        name=Environment.DEVELOPMENT,
        debug=True,
        log_level="DEBUG",
        log_json=False,
        auth_enabled=False,
        rate_limit_enabled=False,
        cors_origins=["*"],
        workers=1,
        reload=True,
    ),
    Environment.STAGING: EnvironmentProfile(
        name=Environment.STAGING,
        debug=False,
        log_level="INFO",
        log_json=True,
        auth_enabled=True,
        rate_limit_enabled=True,
        cors_origins=["https://staging.phoenix-ml.com"],
        workers=2,
        reload=False,
    ),
    Environment.PRODUCTION: EnvironmentProfile(
        name=Environment.PRODUCTION,
        debug=False,
        log_level="WARNING",
        log_json=True,
        auth_enabled=True,
        rate_limit_enabled=True,
        cors_origins=["https://phoenix-ml.com"],
        workers=4,
        reload=False,
    ),
}


def get_environment_profile(env_name: str = "development") -> EnvironmentProfile:
    """Get configuration profile for the given environment."""
    try:
        env = Environment(env_name.lower().strip())
    except ValueError:
        logger.warning(
            "Unknown environment '%s', falling back to development",
            env_name,
        )
        env = Environment.DEVELOPMENT

    profile = PROFILES[env]
    logger.info("Environment profile: %s (debug=%s)", profile.name.value, profile.debug)
    return profile
