from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings managed by Pydantic.
    Reads from environment variables (e.g., REDIS_URL=...)
    """

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    # App Config
    APP_NAME: str = "Phoenix ML Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Feature Store Config
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # Messaging Config
    KAFKA_URL: str = "localhost:9092"

    # Database Config
    # Default to sqlite for local dev without docker
    DATABASE_URL: str = "sqlite+aiosqlite:///./phoenix.db"

    # Model Config — model-agnostic defaults, override via env vars
    DEFAULT_MODEL_ID: str = ""
    DEFAULT_MODEL_VERSION: str = "v1"
    MODEL_CONFIG_DIR: str = "model_configs"

    # Storage Config — override for Docker / Cloud
    CACHE_DIR: str = "/tmp/phoenix/model_cache"
    ARTIFACT_STORAGE_DIR: str = "/tmp/phoenix/remote_storage"

    # Observability Config
    JAEGER_ENDPOINT: str = "http://localhost:4317"

    # MLflow Config
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
