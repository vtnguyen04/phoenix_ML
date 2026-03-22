"""Prediction cache — hash-based caching for identical prediction requests."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Prometheus metrics
cache_hits = Counter("phoenix_cache_hits_total", "Prediction cache hits")
cache_misses = Counter("phoenix_cache_misses_total", "Prediction cache misses")


class PredictionCache:
    """In-memory prediction cache with TTL.

    Key = hash(model_id + version + sorted features).
    Production upgrade: switch to Redis for cross-instance sharing.
    """

    def __init__(self, default_ttl_seconds: int = 300, max_size: int = 10_000) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl_seconds
        self._max_size = max_size

    @staticmethod
    def _make_key(model_id: str, version: str, features: list[float]) -> str:
        """Create deterministic cache key from prediction inputs."""
        raw = json.dumps(
            {"model_id": model_id, "version": version, "features": features},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self, model_id: str, version: str, features: list[float]
    ) -> dict[str, Any] | None:
        """Look up cached prediction. Returns None on miss or TTL expiry."""
        key = self._make_key(model_id, version, features)
        entry = self._cache.get(key)
        if entry is None:
            cache_misses.inc()
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._cache[key]
            cache_misses.inc()
            return None
        cache_hits.inc()
        return value  # type: ignore[no-any-return]

    def set(
        self,
        model_id: str,
        version: str,
        features: list[float],
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store prediction result in cache."""
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        key = self._make_key(model_id, version, features)
        expires_at = time.monotonic() + (ttl or self._default_ttl)
        self._cache[key] = (result, expires_at)

    def invalidate_model(self, model_id: str) -> int:
        """Remove all cached entries for a model (on model update)."""
        keys_to_remove = [
            k for k, (v, _) in self._cache.items()
            if isinstance(v, dict) and v.get("model_id") == model_id
        ]
        for k in keys_to_remove:
            del self._cache[k]
        if keys_to_remove:
            logger.info("Invalidated %d cache entries for model %s", len(keys_to_remove), model_id)
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()

    def _evict_oldest(self) -> None:
        """Remove oldest 10% of entries when cache is full."""
        items = sorted(self._cache.items(), key=lambda x: x[1][1])
        evict_count = max(1, len(items) // 10)
        for key, _ in items[:evict_count]:
            del self._cache[key]

    @property
    def size(self) -> int:
        return len(self._cache)
