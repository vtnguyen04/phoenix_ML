"""
DataLoader Registry — Unified plugin resolution via Chain-of-Responsibility.

Resolves the correct DataLoader for a model_id through a resolver chain:
    1. ProgrammaticResolver  — Explicit registration via code
    2. ConfigResolver         — YAML config: data_loader class path
    3. TaskTypeResolver       — YAML config: task_type → default loader

Custom loaders can be plugged in via:
    1. YAML: `data_loader: my_package.MyLoader`
    2. Code: `DataLoaderRegistry.register("my-model", MyLoader)`
"""

import importlib
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from src.domain.training.services.data_loader_plugin import IDataLoader

logger = logging.getLogger(__name__)


# ─── Task-type → Loader mapping (OCP: extend by adding entries) ──

_TASK_TYPE_LOADERS: dict[str, str] = {
    "classification": "src.infrastructure.data_loaders.tabular_loader.TabularDataLoader",
    "regression": "src.infrastructure.data_loaders.tabular_loader.TabularDataLoader",
    "image_classification": "src.infrastructure.data_loaders.image_loader.ImageDataLoader",
}


# ─── Programmatic Registry ───────────────────────────────────────


class DataLoaderRegistry:
    """Plugin registry for DataLoaders (Strategy pattern).

    Allows code-level registration of custom loaders per model_id.
    """

    _registry: dict[str, type[IDataLoader]] = {}

    @classmethod
    def register(cls, model_id: str, loader_cls: type[IDataLoader]) -> None:
        """Register a custom DataLoader for a specific model."""
        cls._registry[model_id] = loader_cls
        logger.info("Registered DataLoader %s → %s", model_id, loader_cls.__name__)

    @classmethod
    def get(cls, model_id: str) -> type[IDataLoader] | None:
        """Get a registered DataLoader by model_id."""
        return cls._registry.get(model_id)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._registry.clear()


# ─── Chain-of-Responsibility Resolvers ────────────────────────────


class _Resolver(ABC):
    """Base class for DataLoader resolvers (Chain-of-Responsibility)."""

    @abstractmethod
    def resolve(self, model_id: str, task_type: str, loader_path: str) -> IDataLoader | None:
        """Try to resolve a DataLoader. Return None to pass to next resolver."""


class _ProgrammaticResolver(_Resolver):
    """Resolve from explicit code registration."""

    def resolve(self, model_id: str, task_type: str, loader_path: str) -> IDataLoader | None:
        registered = DataLoaderRegistry.get(model_id)
        if registered is not None:
            logger.debug("Resolved via registry: %s → %s", model_id, registered.__name__)
            return registered()
        return None


class _ConfigClassResolver(_Resolver):
    """Resolve from YAML config data_loader class path."""

    def resolve(self, model_id: str, task_type: str, loader_path: str) -> IDataLoader | None:
        if not loader_path:
            return None
        try:
            loader_cls = _import_class(loader_path)
            logger.info("Resolved via config: %s → %s", model_id, loader_path)
            return loader_cls()
        except Exception as e:
            logger.warning("Failed to load '%s': %s", loader_path, e)
            return None


class _TaskTypeResolver(_Resolver):
    """Resolve from task_type → default loader mapping."""

    def resolve(self, model_id: str, task_type: str, loader_path: str) -> IDataLoader | None:
        default_path = _TASK_TYPE_LOADERS.get(task_type)
        if not default_path:
            return None
        try:
            loader_cls = _import_class(default_path)
            logger_cls = loader_cls.__name__
            logger.debug(
                "Resolved via task_type: %s (%s) → %s",
                model_id, task_type, logger_cls,
            )
            return loader_cls()
        except Exception as e:
            logger.warning("Failed default for task_type=%s: %s", task_type, e)
            return None


# Ordered resolver chain
_RESOLVER_CHAIN: list[_Resolver] = [
    _ProgrammaticResolver(),
    _ConfigClassResolver(),
    _TaskTypeResolver(),
]


# ─── Public API ───────────────────────────────────────────────────


def _import_class(dotted_path: str) -> type[IDataLoader]:
    """Import a class from a dotted module path."""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, IDataLoader)):
        raise TypeError(f"{dotted_path} is not a subclass of IDataLoader")
    return cls


def _load_config_fields(model_id: str) -> tuple[str, str]:
    """Load task_type and data_loader path from model config.

    Uses ModelConfigLoader (single source of truth for YAML parsing).
    Returns (task_type, data_loader_class_path).
    """
    from src.infrastructure.bootstrap.model_config_loader import (  # noqa: PLC0415
        load_model_config,
    )

    config_dir = Path(os.environ.get("MODEL_CONFIG_DIR", "model_configs"))
    config_path = config_dir / f"{model_id}.yaml"

    if not config_path.exists():
        return "classification", ""

    try:
        config = load_model_config(config_path)
        # data_loader comes from raw YAML, not ModelConfig VO
        # Read separately since it's infrastructure-only concern

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return config.task_type, raw.get("data_loader", "")
    except Exception as e:
        logger.warning("Failed to read config for %s: %s", model_id, e)
        return "classification", ""


def resolve_data_loader(model_id: str) -> IDataLoader:
    """Resolve the correct DataLoader for a model_id.

    Walks the resolver chain (Chain-of-Responsibility):
        1. ProgrammaticResolver  — code-registered loaders
        2. ConfigClassResolver   — YAML data_loader field
        3. TaskTypeResolver      — task_type → default mapping
        4. Ultimate fallback     — TabularDataLoader

    Returns:
        Instantiated IDataLoader ready to use.
    """
    task_type, loader_path = _load_config_fields(model_id)

    for resolver in _RESOLVER_CHAIN:
        result = resolver.resolve(model_id, task_type, loader_path)
        if result is not None:
            return result

    # Ultimate fallback (avoids circular import)
    from src.infrastructure.data_loaders.tabular_loader import (  # noqa: PLC0415
        TabularDataLoader,
    )

    logger.debug("Fallback: %s → TabularDataLoader", model_id)
    return TabularDataLoader()
