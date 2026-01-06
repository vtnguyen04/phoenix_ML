from abc import ABC, abstractmethod


class FeatureStore(ABC):
    """
    Interface for retrieving online features.
    Adheres to DIP: High-level modules depend on this abstraction.
    """
    
    @abstractmethod
    async def get_online_features(
        self, 
        entity_id: str, 
        feature_names: list[str]
    ) -> list[float] | None:
        """
        Retrieve specific features for an entity (user, item, etc.).
        Returns None if entity not found.
        """
        pass

    @abstractmethod
    async def add_features(self, entity_id: str, data: dict[str, float]) -> None:
        """
        Add or update features for an entity.
        Useful for seeding data or real-time ingestion.
        """
        pass
