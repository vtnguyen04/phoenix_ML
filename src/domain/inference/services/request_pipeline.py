import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, cast

from src.domain.inference.services.inference_service import PredictionRequest

logger = logging.getLogger(__name__)


class PipelineHandler(ABC):
    """
    Base handler for Chain of Responsibility pattern.
    Each handler processes the request and optionally passes it to the next.
    """

    def __init__(self) -> None:
        self._next: PipelineHandler | None = None

    def set_next(self, handler: "PipelineHandler") -> "PipelineHandler":
        self._next = handler
        return handler

    @abstractmethod
    async def handle(self, request: PredictionRequest) -> PredictionRequest:
        if self._next:
            return await self._next.handle(request)
        return request


class ValidationHandler(PipelineHandler):
    """Validates that the prediction request has required fields."""

    async def handle(self, request: PredictionRequest) -> PredictionRequest:
        if not request.model_id:
            raise ValueError("model_id is required")

        has_features = request.features is not None and len(request.features) > 0
        has_entity = request.entity_id is not None

        if not has_features and not has_entity:
            raise ValueError("Either features or entity_id must be provided")

        return await super().handle(request)


class RateLimitHandler(PipelineHandler):
    """Simple token-bucket rate limiter per client/entity."""

    def __init__(self, max_requests: int = 100) -> None:
        super().__init__()
        self._max_requests = max_requests
        self._request_counts: dict[str, int] = {}

    async def handle(self, request: PredictionRequest) -> PredictionRequest:
        client_key = request.entity_id or "anonymous"
        count = self._request_counts.get(client_key, 0)

        if count >= self._max_requests:
            raise PermissionError(
                f"Rate limit exceeded for {client_key}: {self._max_requests} requests"
            )

        self._request_counts[client_key] = count + 1
        return await super().handle(request)

    def reset(self) -> None:
        self._request_counts.clear()


class CacheHandler(PipelineHandler):
    """Checks an in-memory cache before processing."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[str, Any] = {}

    async def handle(self, request: PredictionRequest) -> PredictionRequest:
        cache_key = f"{request.model_id}:{request.model_version}:{request.features}"
        cached = self._cache.get(cache_key)

        if cached is not None:
            logger.info("Cache hit for %s", cache_key)
            return cast(PredictionRequest, cached)

        result = await super().handle(request)
        self._cache[cache_key] = result
        return result

    def clear(self) -> None:
        self._cache.clear()


class FeatureEnrichmentHandler(PipelineHandler):
    """Resolves features from feature store if not provided in the request."""

    def __init__(self, feature_store: Any) -> None:
        super().__init__()
        self._feature_store = feature_store

    async def handle(self, request: PredictionRequest) -> PredictionRequest:
        if request.features is not None:
            return await super().handle(request)

        if not request.entity_id:
            return await super().handle(request)

        features = await self._feature_store.get_online_features(
            request.entity_id, ["f1", "f2", "f3", "f4"]
        )

        if features:
            request = replace(request, features=features)

        return await super().handle(request)


def build_pipeline(
    feature_store: Any | None = None,
    max_requests: int = 100,
) -> PipelineHandler:
    """Factory function to build the standard request pipeline chain."""
    validation = ValidationHandler()
    rate_limit = RateLimitHandler(max_requests=max_requests)
    cache = CacheHandler()

    validation.set_next(rate_limit).set_next(cache)

    if feature_store:
        enrichment = FeatureEnrichmentHandler(feature_store)
        cache.set_next(enrichment)

    return validation
