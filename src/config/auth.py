"""Authentication and security settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """JWT, API-key, and rate-limit configuration."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # JWT
    JWT_SECRET_KEY: str = "phoenix-ml-dev-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_MINUTES: int = 1440  # 24h

    # API Key (simpler auth for service-to-service)
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: str = ""  # comma-separated valid keys, empty = disabled

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100  # per IP
    RATE_LIMIT_PREDICT_PER_MINUTE: int = 60  # /predict stricter
    RATE_LIMIT_BURST: int = 20  # burst allowance

    # Auth toggle (disable for dev/testing)
    AUTH_ENABLED: bool = False
