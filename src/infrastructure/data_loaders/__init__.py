# ruff: noqa: F401
"""
Infrastructure DataLoaders — Concrete implementations of IDataLoader.

Provides dataset loading for all registered model types.
New models can register custom loaders via model_configs YAML
(`data_loader` field) or fall back to the default CSV/NPZ loader.

Architecture:
    domain/training/services/data_loader_plugin.py  ← IDataLoader ABC
    infrastructure/data_loaders/                     ← THIS PACKAGE
        ├── registry.py          ← Unified plugin registry
        ├── tabular_loader.py    ← CSV/Parquet loader (default)
        ├── image_loader.py      ← NPZ/image-dir loader
        └── <custom>_loader.py   ← User-defined loaders
"""

from src.infrastructure.data_loaders.image_loader import ImageDataLoader
from src.infrastructure.data_loaders.registry import (
    DataLoaderRegistry,
    resolve_data_loader,
)
from src.infrastructure.data_loaders.tabular_loader import TabularDataLoader

__all__ = [
    "DataLoaderRegistry",
    "ImageDataLoader",
    "TabularDataLoader",
    "resolve_data_loader",
]
