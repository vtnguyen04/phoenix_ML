from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings managed by Pydantic.
    Reads from environment variables (e.g., REDIS_URL=...)
    """
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

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

    # Model Config
    DEFAULT_MODEL_PATH: str = "local://models/demo.onnx"

@lru_cache
def get_settings() -> Settings:
    return Settings()
