class PhoenixBaseError(Exception):
    """Base exception for the Phoenix ML Platform."""

    def __init__(self, message: str, code: str | None = None) -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)


class ModelNotFoundError(PhoenixBaseError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_id: str, version: str | None = None) -> None:
        detail = f"Model '{model_id}'"
        if version:
            detail += f" version '{version}'"
        super().__init__(f"{detail} not found", code="MODEL_NOT_FOUND")


class FeatureStoreError(PhoenixBaseError):
    """Raised when feature retrieval fails."""

    def __init__(self, entity_id: str, reason: str = "unknown") -> None:
        super().__init__(
            f"Failed to fetch features for entity '{entity_id}': {reason}",
            code="FEATURE_STORE_ERROR",
        )


class InferenceError(PhoenixBaseError):
    """Raised when model inference fails."""

    def __init__(self, model_id: str, reason: str = "unknown") -> None:
        super().__init__(
            f"Inference failed for model '{model_id}': {reason}",
            code="INFERENCE_ERROR",
        )


class CircuitBreakerOpenError(PhoenixBaseError):
    """Raised when the circuit breaker is in OPEN state."""

    def __init__(self, service_name: str = "inference") -> None:
        super().__init__(
            f"Circuit breaker is OPEN for service '{service_name}'",
            code="CIRCUIT_BREAKER_OPEN",
        )


class RateLimitExceededError(PhoenixBaseError):
    """Raised when a client exceeds the rate limit."""

    def __init__(self, client_id: str, limit: int) -> None:
        super().__init__(
            f"Rate limit exceeded for client '{client_id}': max {limit} requests",
            code="RATE_LIMIT_EXCEEDED",
        )


class ValidationError(PhoenixBaseError):
    """Raised when request validation fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Validation failed: {detail}", code="VALIDATION_ERROR")
