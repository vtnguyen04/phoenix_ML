"""Phoenix ML Platform — Framework Core.

Public API for the Phoenix ML framework:

    from phoenix_ml import PhoenixPlatform

    platform = PhoenixPlatform(
        database_url="postgresql+asyncpg://user:pass@cloud-db:5432/mydb",
        redis_url="redis://cloud-redis:6379",
        mlflow_uri="http://cloud-mlflow:5000",
        model_configs_dir="./my_models/",
    )
    platform.serve(host="0.0.0.0", port=8000)
"""

__version__ = "0.1.0"

# ── Public API ────────────────────────────────────────────────────
from phoenix_ml.config import Settings, get_settings
from phoenix_ml.domain.inference.value_objects.model_config import ModelConfig
from phoenix_ml.platform import PhoenixPlatform

__all__ = [
    "ModelConfig",
    "PhoenixPlatform",
    "Settings",
    "__version__",
    "get_settings",
]
