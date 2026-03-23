"""Command DTO for batch prediction requests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPredictCommand:
    """Command to perform batch predictions.

    Attributes:
        model_id: Target model identifier.
        model_version: Optional specific version.
        batch: List of feature vectors (each is a list of floats).
        entity_ids: Optional list of entity IDs for feature store lookup.
    """

    model_id: str
    batch: list[list[float]]
    model_version: str | None = None
    entity_ids: list[str] | None = None
