"""Registry for model-specific plugin implementations.

Maps ``model_id`` to concrete instances of ``ITrainer``, ``IDataLoader``,
``IPreprocessor``, ``IPostprocessor``, and ``MetricCalculator``.

Lookup:
    ``get_<plugin>(model_id)`` returns the registered instance or a default.
    Defaults: ``PassthroughPreprocessor``, ``ClassificationPostprocessor``,
    ``_DefaultTrainer`` (raises ``NotImplementedError`` on ``train()``).
    ``IDataLoader`` and ``MetricCalculator`` return ``None`` if unregistered.
"""

import logging
from collections.abc import Callable
from typing import Any

from phoenix_ml.domain.inference.services.processor_plugin import (
    ClassificationPostprocessor,
    IPostprocessor,
    IPreprocessor,
    PassthroughPreprocessor,
)
from phoenix_ml.domain.training.services.data_loader_plugin import IDataLoader
from phoenix_ml.domain.training.services.trainer_plugin import ITrainer, TrainResult

logger = logging.getLogger(__name__)

# Type alias for custom metric calculators
MetricCalculator = Callable[[Any, Any], dict[str, float]]


class _DefaultTrainer(ITrainer):
    """Fallback trainer that raises a helpful error."""

    async def train(self, config: dict[str, Any]) -> TrainResult:
        msg = (
            "No trainer registered for this model. "
            "Implement ITrainer and register it with the PluginRegistry."
        )
        raise NotImplementedError(msg)

    async def validate(self, model_path: str, data_path: str) -> dict[str, float]:
        return {}


class PluginRegistry:
    """Central registry for model-specific plugin implementations.

    Each model can have its own set of plugins for training,
    data loading, preprocessing, postprocessing, and metric
    calculation. When a plugin is not registered for a specific
    model, sensible defaults are used.
    """

    def __init__(self) -> None:
        self._trainers: dict[str, ITrainer] = {}
        self._data_loaders: dict[str, IDataLoader] = {}
        self._preprocessors: dict[str, IPreprocessor] = {}
        self._postprocessors: dict[str, IPostprocessor] = {}
        self._metric_calculators: dict[str, MetricCalculator] = {}

        # Defaults for models without registered plugins
        self._default_preprocessor = PassthroughPreprocessor()
        self._default_postprocessor = ClassificationPostprocessor()
        self._default_trainer = _DefaultTrainer()

    # --- Trainer ---
    def register_trainer(self, model_id: str, trainer: ITrainer) -> None:
        """Register a trainer for a specific model."""
        self._trainers[model_id] = trainer
        logger.info("📌 Registered trainer for model '%s': %s", model_id, type(trainer).__name__)

    def get_trainer(self, model_id: str) -> ITrainer:
        """Get the trainer for a model, or default."""
        return self._trainers.get(model_id, self._default_trainer)

    def has_trainer(self, model_id: str) -> bool:
        """Check if a custom trainer is registered for a model."""
        return model_id in self._trainers

    # --- DataLoader ---
    def register_data_loader(self, model_id: str, loader: IDataLoader) -> None:
        """Register a data loader for a specific model."""
        self._data_loaders[model_id] = loader
        logger.info(
            "📌 Registered data_loader for '%s': %s",
            model_id,
            type(loader).__name__,
        )

    def get_data_loader(self, model_id: str) -> IDataLoader | None:
        """Get data loader for a model, or None if not registered."""
        return self._data_loaders.get(model_id)

    def has_data_loader(self, model_id: str) -> bool:
        """Check if a custom data loader is registered for a model."""
        return model_id in self._data_loaders

    # --- Preprocessor ---
    def register_preprocessor(self, model_id: str, preprocessor: IPreprocessor) -> None:
        """Register a preprocessor for a specific model."""
        self._preprocessors[model_id] = preprocessor
        logger.info(
            "📌 Registered preprocessor for '%s': %s",
            model_id,
            type(preprocessor).__name__,
        )

    def get_preprocessor(self, model_id: str) -> IPreprocessor:
        """Get preprocessor for a model, or PassthroughPreprocessor."""
        return self._preprocessors.get(model_id, self._default_preprocessor)

    # --- Postprocessor ---
    def register_postprocessor(self, model_id: str, postprocessor: IPostprocessor) -> None:
        """Register a postprocessor for a specific model."""
        self._postprocessors[model_id] = postprocessor
        logger.info(
            "📌 Registered postprocessor for '%s': %s",
            model_id,
            type(postprocessor).__name__,
        )

    def get_postprocessor(self, model_id: str) -> IPostprocessor:
        """Get postprocessor for a model, or ClassificationPostprocessor."""
        return self._postprocessors.get(model_id, self._default_postprocessor)

    # --- Metric Calculator ---
    def register_metric_calculator(
        self,
        model_id: str,
        calculator: MetricCalculator,
    ) -> None:
        """Register a custom metric calculator for a model.

        The calculator receives (y_true, y_pred) and returns a
        dict[str, float] of metric_name → value.

        Example::

            def my_metrics(y_true, y_pred):
                return {"custom_score": compute_custom(y_true, y_pred)}

            registry.register_metric_calculator("my-model", my_metrics)
        """
        self._metric_calculators[model_id] = calculator
        logger.info("📌 Registered metric_calculator for '%s'", model_id)

    def get_metric_calculator(self, model_id: str) -> MetricCalculator | None:
        """Get custom metric calculator for a model."""
        return self._metric_calculators.get(model_id)

    def has_metric_calculator(self, model_id: str) -> bool:
        """Check if a custom metric calculator is registered."""
        return model_id in self._metric_calculators

    # --- Introspection ---
    @property
    def registered_models(self) -> list[str]:
        """List all model IDs with at least one registered plugin."""
        models: set[str] = set()
        models.update(self._trainers.keys())
        models.update(self._data_loaders.keys())
        models.update(self._preprocessors.keys())
        models.update(self._postprocessors.keys())
        models.update(self._metric_calculators.keys())
        return sorted(models)

    def summary(self) -> dict[str, dict[str, str]]:
        """Return a summary of all registered plugins per model."""
        result: dict[str, dict[str, str]] = {}
        for model_id in self.registered_models:
            result[model_id] = {
                "trainer": type(self.get_trainer(model_id)).__name__,
                "data_loader": type(dl).__name__ if (dl := self.get_data_loader(model_id)) else "—",
                "preprocessor": type(self.get_preprocessor(model_id)).__name__,
                "postprocessor": type(self.get_postprocessor(model_id)).__name__,
                "metric_calculator": "✓" if self.has_metric_calculator(model_id) else "—",
            }
        return result

