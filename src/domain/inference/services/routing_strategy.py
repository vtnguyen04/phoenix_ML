import logging
import random
from abc import ABC, abstractmethod
from typing import Any

from src.domain.inference.entities.model import Model

logger = logging.getLogger(__name__)


class RoutingStrategy(ABC):
    """Strategy Pattern for determining which model to use for a request."""

    @abstractmethod
    def select_model(
        self, models: list[Model], context: dict[str, Any] | None = None
    ) -> Model:
        pass


class SingleModelStrategy(RoutingStrategy):
    """Default strategy: Returns the first available active model."""

    def select_model(
        self, models: list[Model], context: dict[str, Any] | None = None
    ) -> Model:
        active_models = [m for m in models if m.is_active]
        if not active_models:
            raise ValueError("No active models available")
        return active_models[0]


class ABTestStrategy(RoutingStrategy):
    """Routes traffic based on a percentage split (e.g., 90/10)."""

    def __init__(self, challenger_traffic_percentage: float = 0.1):
        if not 0 <= challenger_traffic_percentage <= 1:
            raise ValueError("Traffic percentage must be between 0 and 1")
        self.challenger_percentage = challenger_traffic_percentage

    def select_model(
        self, models: list[Model], context: dict[str, Any] | None = None
    ) -> Model:
        MIN_MODELS_FOR_AB = 2
        if len(models) < MIN_MODELS_FOR_AB:
            return models[0]

        champion = models[0]
        challenger = models[-1]

        if random.random() < self.challenger_percentage:
            return challenger
        return champion


class CanaryStrategy(RoutingStrategy):
    """Routes a small % of traffic to a challenger for gradual rollout."""

    MAX_PERCENTAGE = 100

    def __init__(self, canary_percentage: float = 5.0):
        if not 0 <= canary_percentage <= self.MAX_PERCENTAGE:
            raise ValueError("Canary percentage must be between 0 and 100")
        self.canary_percentage = canary_percentage

    def select_model(
        self, models: list[Model], context: dict[str, Any] | None = None
    ) -> Model:
        champion = self._find_by_role(models, "champion")
        challenger = self._find_by_role(models, "challenger")

        if challenger and random.random() < self.canary_percentage / 100:
            logger.info("Canary routing: selected challenger %s", challenger.unique_key)
            return challenger
        return champion

    def _find_by_role(self, models: list[Model], role: str) -> Model | None:
        for m in models:
            m_meta = m.metadata or {}
            if m_meta.get("role") == role:
                return m
        return models[0] if role == "champion" else None


class ShadowStrategy(RoutingStrategy):
    """
    Routes to champion model and logs the challenger model selection
    for shadow comparison. The actual shadow prediction is handled
    by the caller (InferenceService).
    """

    def __init__(self) -> None:
        self._shadow_model: Model | None = None

    @property
    def shadow_model(self) -> Model | None:
        return self._shadow_model

    def select_model(
        self, models: list[Model], context: dict[str, Any] | None = None
    ) -> Model:
        champion = self._find_by_role(models, "champion")
        self._shadow_model = self._find_by_role(models, "challenger")

        if self._shadow_model:
            logger.info(
                "Shadow routing: champion=%s, shadow=%s",
                champion.unique_key,
                self._shadow_model.unique_key,
            )
        return champion

    def _find_by_role(self, models: list[Model], role: str) -> Model | None:
        for m in models:
            m_meta = m.metadata or {}
            if m_meta.get("role") == role:
                return m
        return models[0] if role == "champion" else None
