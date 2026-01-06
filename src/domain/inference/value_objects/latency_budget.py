from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyBudget:
    """
    Value Object representing the time budget for inference.
    """
    max_latency_ms: float
    
    def __post_init__(self) -> None:
        if self.max_latency_ms <= 0:
            raise ValueError("Latency budget must be positive")

    def is_exceeded_by(self, latency_ms: float) -> bool:
        return latency_ms > self.max_latency_ms
