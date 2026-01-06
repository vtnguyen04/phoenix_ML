
from src.domain.feature_store.repositories.feature_store import FeatureStore


class InMemoryFeatureStore(FeatureStore):
    """
    In-memory implementation of FeatureStore for testing and local development.
    """
    
    def __init__(self) -> None:
        # Mock data storage: entity_id -> {feature_name: value}
        self._store: dict[str, dict[str, float]] = {}

    async def get_online_features(
        self, 
        entity_id: str, 
        feature_names: list[str]
    ) -> list[float] | None:
        if entity_id not in self._store:
            return None
        
        features = []
        entity_data = self._store[entity_id]
        
        for name in feature_names:
            # Default to 0.0 if feature missing, 
            # or handle strict error depending on requirements
            features.append(entity_data.get(name, 0.0))
            
        return features

    async def add_features(self, entity_id: str, data: dict[str, float]) -> None:
        """Helper method to seed data for testing"""
        self._store[entity_id] = data
