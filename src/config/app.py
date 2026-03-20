"""App-level settings — name, version, debug mode."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Core application identity and behaviour."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    APP_NAME: str = "Phoenix ML Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Model defaults
    DEFAULT_MODEL_ID: str = ""
    DEFAULT_MODEL_VERSION: str = "v1"
    DEFAULT_TASK_TYPE: str = "classification"
    MODEL_CONFIG_DIR: str = "model_configs"
