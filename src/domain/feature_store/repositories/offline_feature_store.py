from abc import ABC, abstractmethod
from typing import Any


class OfflineFeatureStore(ABC):
    """
    Interface for retrieving offline/historical features, used for distributed
    training or batch inference. Adheres to ISP (Interface Segregation Principle).
    """

    @abstractmethod
    async def get_historical_features(
        self, entity_ids: list[str], feature_names: list[str]
    ) -> list[dict[str, Any]]:
        """
        Retrieve point-in-time historical features for a batch of entities.
        """
        pass

    @abstractmethod
    async def extract_training_dataset(self, query: str, output_path: str) -> None:
        """
        Extract a materialized training dataset from the offline store.
        """
        pass
