
import redis.asyncio as redis

from src.domain.feature_store.repositories.feature_store import FeatureStore





class RedisFeatureStore(FeatureStore):

    """

    Production implementation of FeatureStore using Redis.

    """



    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:

        self.redis = redis.from_url(redis_url, decode_responses=True)  # type: ignore



    async def get_online_features(


        self, 
        entity_id: str, 
        feature_names: list[str]
    ) -> list[float] | None:
        # Assumption: Features are stored as a hash map or JSON string in Redis
        # Key format: "features:{entity_id}"
        key = f"features:{entity_id}"
        
        # Strategy: Use HMGET for Hash storage
        try:
            values = await self.redis.hmget(key, feature_names)
            
            # Check if key exists (Redis returns list of Nones if key doesn't exist)
            if all(v is None for v in values):
                return None
            
            # Convert to floats, handling missing individual features
            return [float(v) if v is not None else 0.0 for v in values]
            
        except Exception:
            # Fallback or log error
            # In production, you might want to return None or raise specific DomainError
            return None
