"""Phoenix ML Platform — Framework Core.

Public API surface for the Phoenix ML framework. Import key components:

    from src import get_settings, ModelConfig, run_server

For framework usage:
    from src.config import get_settings
    from src.domain.inference.value_objects.model_config import ModelConfig
    from src.infrastructure.http.fastapi_server import app, run
"""

__version__ = "0.1.0"

# ── Public API ────────────────────────────────────────────────────
from src.config import Settings, get_settings
from src.domain.inference.value_objects.model_config import ModelConfig
from src.infrastructure.http.fastapi_server import run as run_server

__all__ = [
    "ModelConfig",
    "Settings",
    "__version__",
    "get_settings",
    "run_server",
]
