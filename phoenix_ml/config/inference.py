"""Inference & batch processing settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    """Engine, batch, and confidence configuration."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Engine selection
    INFERENCE_ENGINE: str = "onnx"
    TRITON_URL: str = "http://localhost:8000"
    CONFIDENCE_THRESHOLD: float = 0.5

    # Batch processing
    BATCH_MAX_SIZE: int = 16
    BATCH_MAX_WAIT_MS: int = 10

    # Storage
    CACHE_DIR: str = "/tmp/phoenix/model_cache"
    ARTIFACT_STORAGE_DIR: str = "/tmp/phoenix/remote_storage"
