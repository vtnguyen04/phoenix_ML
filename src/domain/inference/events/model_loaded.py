from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ModelLoaded:
    """
    Domain Event triggered when a model is successfully loaded into memory/GPU.
    """
    model_id: str
    model_version: str
    framework: str
    occurred_at: datetime = datetime.now(UTC)
