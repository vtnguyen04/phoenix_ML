import random
from abc import ABC, abstractmethod
from typing import Any

from src.domain.inference.entities.model import Model


class RoutingStrategy(ABC):
    """
    Strategy Pattern for determining which model to use for a request.
    """
    @abstractmethod
    def select_model(
        self, 
        models: list[Model], 
        context: dict[str, Any] | None = None
    ) -> Model:
        pass


class SingleModelStrategy(RoutingStrategy):
    """
    Default strategy: Returns the first available active model.
    """
    def select_model(
        self, 
        models: list[Model], 
        context: dict[str, Any] | None = None
    ) -> Model:
        active_models = [m for m in models if m.is_active]
        if not active_models:
            raise ValueError("No active models available")
        return active_models[0]


class ABTestStrategy(RoutingStrategy):
    """
    Routes traffic based on a percentage split (e.g., 90% Champion, 10% Challenger).
    """
    def __init__(self, challenger_traffic_percentage: float = 0.1):
        if not 0 <= challenger_traffic_percentage <= 1:
            raise ValueError("Traffic percentage must be between 0 and 1")
        self.challenger_percentage = challenger_traffic_percentage

    def select_model(
        self, 
        models: list[Model], 
        context: dict[str, Any] | None = None
    ) -> Model:
        # Simplification: Assume list has [Champion, Challenger] 
        # or we identify them by metadata
        # For this prototype, we assume the last model in the list is the Challenger
        MIN_MODELS_FOR_AB = 2
        if len(models) < MIN_MODELS_FOR_AB:
            return models[0]
        
        champion = models[0]
        challenger = models[-1]

        if random.random() < self.challenger_percentage:
            return challenger
        return champion