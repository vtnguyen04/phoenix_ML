from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ModelLoaded:
    """Domain event emitted when a model is loaded into memory."""

    model_id: str
    model_version: str
    framework: str
    occurred_at: datetime = datetime.now(UTC)
