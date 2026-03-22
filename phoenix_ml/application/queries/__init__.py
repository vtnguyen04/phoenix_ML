from dataclasses import dataclass


@dataclass(frozen=True)
class GetModelQuery:
    """Query to retrieve a model by ID and version."""

    model_id: str
    version: str | None = None


@dataclass(frozen=True)
class GetDriftReportQuery:
    """Query to retrieve drift reports for a model."""

    model_id: str
    limit: int = 10


@dataclass(frozen=True)
class GetPredictionLogsQuery:
    """Query to retrieve prediction logs for a model."""

    model_id: str
    limit: int = 100
    feature_index: int | None = None


@dataclass(frozen=True)
class GetModelPerformanceQuery:
    """Query to retrieve model performance metrics."""

    model_id: str
    version: str | None = None
