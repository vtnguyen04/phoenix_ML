"""
Centralized configuration — split by domain concern.

Each sub-module defines one Settings class.
This __init__.py composes them via multiple inheritance so that:
  - settings.REDIS_URL        → works with full mypy type safety
  - settings.BATCH_MAX_SIZE   → same
  - Each domain config file is self-contained and easy to find

Usage:
    from src.config import get_settings
    settings = get_settings()
    settings.APP_NAME           # from AppSettings
    settings.BATCH_MAX_SIZE     # from InferenceSettings
    settings.DRIFT_PSI_MODERATE # from MonitoringSettings
    settings.KAFKA_URL          # from InfrastructureSettings
"""

from functools import lru_cache

from src.config.app import AppSettings
from src.config.inference import InferenceSettings
from src.config.infrastructure import InfrastructureSettings
from src.config.monitoring import MonitoringSettings


class Settings(AppSettings, InferenceSettings, MonitoringSettings, InfrastructureSettings):
    """Flat composed settings — all attributes type-safe, split by domain file."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
