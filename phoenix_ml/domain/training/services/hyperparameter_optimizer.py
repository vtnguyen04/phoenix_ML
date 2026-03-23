"""Hyperparameter search over a defined parameter space.

Generates ``TrainingConfig`` variants using grid search or random
search strategies. Implements the Strategy pattern via ``SearchStrategy``.
"""

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from phoenix_ml.domain.training.entities.training_config import TrainingConfig


@dataclass(frozen=True)
class SearchSpace:
    """Defines the hyperparameter search space."""

    learning_rates: list[float]
    batch_sizes: list[int]
    epochs_options: list[int]
    extra: dict[str, list[str]]


class SearchStrategy(ABC):
    """Strategy interface for hyperparameter search algorithms."""

    @abstractmethod
    def generate_configs(
        self, base_config: TrainingConfig, space: SearchSpace, max_trials: int
    ) -> list[TrainingConfig]:
        """Generate a list of configs to try."""
        ...


class GridSearchStrategy(SearchStrategy):
    """Exhaustive grid search over all hyperparameter combinations."""

    def generate_configs(
        self, base_config: TrainingConfig, space: SearchSpace, max_trials: int
    ) -> list[TrainingConfig]:
        configs: list[TrainingConfig] = []

        combinations = list(
            itertools.product(
                space.learning_rates,
                space.batch_sizes,
                space.epochs_options,
            )
        )

        for lr, bs, ep in combinations[:max_trials]:
            configs.append(
                TrainingConfig(
                    dataset_path=base_config.dataset_path,
                    model_type=base_config.model_type,
                    epochs=ep,
                    batch_size=bs,
                    learning_rate=lr,
                    validation_split=base_config.validation_split,
                    early_stopping_patience=base_config.early_stopping_patience,
                    random_seed=base_config.random_seed,
                    hyperparameters=base_config.hyperparameters,
                )
            )
        return configs


class RandomSearchStrategy(SearchStrategy):
    """Random sampling from the hyperparameter space."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate_configs(
        self, base_config: TrainingConfig, space: SearchSpace, max_trials: int
    ) -> list[TrainingConfig]:
        configs: list[TrainingConfig] = []
        for _ in range(max_trials):
            lr = self._rng.choice(space.learning_rates)
            bs = self._rng.choice(space.batch_sizes)
            ep = self._rng.choice(space.epochs_options)

            configs.append(
                TrainingConfig(
                    dataset_path=base_config.dataset_path,
                    model_type=base_config.model_type,
                    epochs=ep,
                    batch_size=bs,
                    learning_rate=lr,
                    validation_split=base_config.validation_split,
                    early_stopping_patience=base_config.early_stopping_patience,
                    random_seed=base_config.random_seed,
                    hyperparameters=base_config.hyperparameters,
                )
            )
        return configs


class HyperparameterOptimizer:
    """
    Facade for hyperparameter optimization.

    Uses a pluggable SearchStrategy (grid or random) to generate
    training configs from a defined search space.
    """

    def __init__(self, strategy: SearchStrategy | None = None) -> None:
        self._strategy = strategy or GridSearchStrategy()

    def generate_trials(
        self,
        base_config: TrainingConfig,
        space: SearchSpace,
        max_trials: int = 20,
    ) -> list[TrainingConfig]:
        """Generate trial configurations using the selected strategy."""
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        return self._strategy.generate_configs(base_config, space, max_trials)
