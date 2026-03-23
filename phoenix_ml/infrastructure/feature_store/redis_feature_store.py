import redis.asyncio as redis

from phoenix_ml.domain.feature_store.repositories.feature_store import FeatureStore


class RedisFeatureStore(FeatureStore):
    """Redis-backed online feature store.

    Stores features as Redis hashes keyed by ``features:<entity_id>``.
    Each hash field is a feature name mapping to a float value.

    Args:
        redis_url: Redis connection URI (default ``redis://localhost:6379``).

    Key format:
        ``features:<entity_id>`` → ``{feature_name: value, ...}``

    TTL:
        Optional per-entity TTL via ``add_features(ttl_seconds=...)``.
        Use ``get_ttl()`` to check remaining TTL.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self.redis = redis.from_url(redis_url, decode_responses=True)

    # ── Single Entity Operations ─────────────────────────────────

    async def get_online_features(
        self, entity_id: str, feature_names: list[str]
    ) -> list[float] | None:
        """Get features for a single entity."""
        key = f"features:{entity_id}"
        try:
            values = await self.redis.hmget(key, feature_names)  # type: ignore[misc]
            if all(v is None for v in values):
                return None
            return [float(v) if v is not None else 0.0 for v in values]
        except Exception:
            return None

    async def add_features(
        self,
        entity_id: str,
        data: dict[str, float],
        ttl_seconds: int | None = None,
    ) -> None:
        """Add features for a single entity, with optional TTL."""
        key = f"features:{entity_id}"
        await self.redis.hset(key, mapping=data)  # type: ignore[misc]
        if ttl_seconds is not None:
            await self.redis.expire(key, ttl_seconds)

    # ── Batch Operations ─────────────────────────────────────────

    async def get_features_batch(
        self,
        entity_ids: list[str],
        feature_names: list[str],
    ) -> dict[str, list[float] | None]:
        """Get features for multiple entities in a single pipeline call."""
        pipe = self.redis.pipeline()
        for entity_id in entity_ids:
            pipe.hmget(f"features:{entity_id}", feature_names)

        results = await pipe.execute()
        output: dict[str, list[float] | None] = {}
        for entity_id, values in zip(entity_ids, results, strict=False):
            if all(v is None for v in values):
                output[entity_id] = None
            else:
                output[entity_id] = [float(v) if v is not None else 0.0 for v in values]
        return output

    async def add_features_batch(
        self,
        entities: dict[str, dict[str, float]],
        ttl_seconds: int | None = None,
    ) -> int:
        """Add features for multiple entities via pipeline. Returns count added."""
        pipe = self.redis.pipeline()
        for entity_id, data in entities.items():
            key = f"features:{entity_id}"
            pipe.hset(key, mapping=data)
            if ttl_seconds is not None:
                pipe.expire(key, ttl_seconds)
        await pipe.execute()
        return len(entities)

    # ── Delete & List ────────────────────────────────────────────

    async def delete_features(self, entity_id: str) -> bool:
        """Delete all features for an entity. Returns True if key existed."""
        key = f"features:{entity_id}"
        result: int = await self.redis.delete(key)
        return result > 0

    async def delete_features_batch(self, entity_ids: list[str]) -> int:
        """Delete features for multiple entities. Returns count deleted."""
        if not entity_ids:
            return 0
        keys = [f"features:{eid}" for eid in entity_ids]
        result: int = await self.redis.delete(*keys)
        return result

    async def list_entities(self, pattern: str = "features:*", limit: int = 100) -> list[str]:
        """List entity IDs with features stored.

        Uses SCAN (non-blocking) instead of KEYS in production.
        """
        entities: list[str] = []
        async for key in self.redis.scan_iter(match=pattern, count=100):
            entity_id = key.removeprefix("features:")
            entities.append(entity_id)
            if len(entities) >= limit:
                break
        return entities

    async def count_entities(self) -> int:
        """Count total entities with stored features."""
        count = 0
        async for _ in self.redis.scan_iter(match="features:*", count=100):
            count += 1
        return count

    # ── Feature Metadata ─────────────────────────────────────────

    async def get_feature_names(self, entity_id: str) -> list[str]:
        """Get all feature field names for an entity."""
        key = f"features:{entity_id}"
        fields: list[str] = await self.redis.hkeys(key)  # type: ignore[misc]
        return list(fields)

    async def get_feature_count(self, entity_id: str) -> int:
        """Get the number of features stored for an entity."""
        key = f"features:{entity_id}"
        result: int = await self.redis.hlen(key)  # type: ignore[misc]
        return result

    async def get_ttl(self, entity_id: str) -> int:
        """Get remaining TTL in seconds (-1 = no TTL, -2 = key missing)."""
        key = f"features:{entity_id}"
        result: int = await self.redis.ttl(key)
        return result

    # ── Health Check ─────────────────────────────────────────────

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            result: bool = await self.redis.ping()  # type: ignore[misc]
            return result
        except Exception:
            return False
