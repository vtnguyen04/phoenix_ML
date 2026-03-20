"""Infrastructure / external service connection settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class InfrastructureSettings(BaseSettings):
    """Database, messaging, observability, and external service configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./phoenix.db"

    # Feature Store
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # Messaging
    KAFKA_URL: str = "localhost:9092"

    # Observability
    JAEGER_ENDPOINT: str = "http://localhost:4317"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # Model Registry Backend
    MODEL_REGISTRY_BACKEND: str = "postgres"

    # Airflow
    AIRFLOW_API_URL: str = "http://localhost:8080/api/v1"
    AIRFLOW_ADMIN_USER: str = "admin"
    AIRFLOW_ADMIN_PASSWORD: str = "admin"
    AIRFLOW_DAG_ID: str = "self_healing_pipeline"

    # gRPC
    GRPC_PORT: int = 50051
    GRPC_MAX_WORKERS: int = 10
