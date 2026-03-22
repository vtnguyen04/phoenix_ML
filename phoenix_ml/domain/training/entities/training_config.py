"""
Training Configuration Value Object.

Immutable configuration for a training run including hyperparameters,
data sources, and resource constraints.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """
    Value Object encapsulating all parameters for a training run.

    Frozen dataclass to enforce immutability — a config cannot change
    once a training job has started.
    """

    dataset_path: str
    model_type: str = "xgboost"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    random_seed: int = 42
    hyperparameters: tuple[tuple[str, str], ...] = ()

    def get_hyperparameter(self, key: str, default: str = "") -> str:
        """Retrieve a hyperparameter value by key."""
        for k, v in self.hyperparameters:
            if k == key:
                return v
        return default

    def with_hyperparameters(self, **kwargs: str) -> "TrainingConfig":
        """Return a new config with added/overridden hyperparameters."""
        existing = dict(self.hyperparameters)
        existing.update(kwargs)
        return TrainingConfig(
            dataset_path=self.dataset_path,
            model_type=self.model_type,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            validation_split=self.validation_split,
            early_stopping_patience=self.early_stopping_patience,
            random_seed=self.random_seed,
            hyperparameters=tuple(existing.items()),
        )
