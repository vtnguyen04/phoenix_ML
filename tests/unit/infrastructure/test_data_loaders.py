"""Tests for DataLoader infrastructure."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from phoenix_ml.domain.training.services.data_loader_plugin import IDataLoader
from phoenix_ml.infrastructure.data_loaders.image_loader import ImageDataLoader
from phoenix_ml.infrastructure.data_loaders.registry import (
    DataLoaderRegistry,
    resolve_data_loader,
)
from phoenix_ml.infrastructure.data_loaders.tabular_loader import TabularDataLoader

# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def csv_dataset(tmp_path: Path) -> Path:
    """Create a simple CSV dataset for testing."""
    df = pd.DataFrame(
        {
            "feature_a": np.random.randn(100).astype(np.float32),
            "feature_b": np.random.randn(100).astype(np.float32),
            "feature_c": np.random.randn(100).astype(np.float32),
            "target": np.random.randint(0, 2, size=100),
        }
    )
    path = tmp_path / "test_dataset.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def npz_dataset(tmp_path: Path) -> Path:
    """Create a simple NPZ dataset for testing."""
    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, size=(200, 784)).astype(np.float32)
    y = rng.randint(0, 10, size=200).astype(int)
    path = tmp_path / "test_images.npz"
    np.savez_compressed(path, X=x, y=y)
    return path


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Clear DataLoader registry before each test."""
    DataLoaderRegistry.clear()


# ─── TabularDataLoader Tests ──────────────────────────────────────


class TestTabularDataLoader:
    """Tests for the generic TabularDataLoader."""

    def test_load_csv(self, csv_dataset: Path) -> None:
        """Should load CSV and return (X, y) + DatasetInfo."""
        loader = TabularDataLoader()
        data, info = asyncio.run(loader.load(str(csv_dataset), target_column="target"))

        x, y = data
        assert x.shape == (100, 3)
        assert y.shape == (100,)
        assert info.num_samples == 100
        assert info.num_features == 3
        assert info.feature_names == ["feature_a", "feature_b", "feature_c"]
        assert info.data_format == "tabular_csv"

    def test_load_with_default_target(self, csv_dataset: Path) -> None:
        """Should use last column as target when not specified."""
        loader = TabularDataLoader()
        data, info = asyncio.run(loader.load(str(csv_dataset)))

        x, y = data
        assert x.shape[1] == 3  # 4 cols - 1 target = 3 features
        assert info.num_features == 3

    def test_load_with_max_samples(self, csv_dataset: Path) -> None:
        """Should limit samples when max_samples specified."""
        loader = TabularDataLoader()
        data, info = asyncio.run(
            loader.load(str(csv_dataset), target_column="target", max_samples=50),
        )

        assert info.num_samples == 50

    def test_load_file_not_found(self) -> None:
        """Should raise FileNotFoundError for missing file."""
        loader = TabularDataLoader()
        with pytest.raises(FileNotFoundError):
            asyncio.run(loader.load("/nonexistent/path.csv"))

    def test_split_stratified(self, csv_dataset: Path) -> None:
        """Should stratify split for classification targets."""
        loader = TabularDataLoader()
        data, _ = asyncio.run(loader.load(str(csv_dataset), target_column="target"))
        train, test = asyncio.run(loader.split(data, test_size=0.2))

        x_train, y_train = train
        x_test, y_test = test
        assert len(x_train) == 80
        assert len(x_test) == 20
        # Both classes should be present in both sets
        assert len(np.unique(y_train)) >= 2
        assert len(np.unique(y_test)) >= 2

    def test_class_labels_detected(self, csv_dataset: Path) -> None:
        """Should auto-detect class labels for classification."""
        loader = TabularDataLoader()
        _, info = asyncio.run(loader.load(str(csv_dataset), target_column="target"))
        assert info.class_labels == ["0", "1"]

    def test_implements_interface(self) -> None:
        """Should be a proper implementation of IDataLoader."""
        assert issubclass(TabularDataLoader, IDataLoader)


# ─── ImageDataLoader Tests ────────────────────────────────────────


class TestImageDataLoader:
    """Tests for the generic ImageDataLoader."""

    def test_load_npz(self, npz_dataset: Path) -> None:
        """Should load NPZ and return (X, y) + DatasetInfo."""
        loader = ImageDataLoader()
        data, info = asyncio.run(loader.load(str(npz_dataset)))

        x, y = data
        assert x.shape == (200, 784)
        assert y.shape == (200,)
        assert info.num_samples == 200
        assert info.num_features == 784
        assert info.data_format == "image_npz"
        assert x.max() <= 1.0  # normalized

    def test_load_npz_no_normalize(self, npz_dataset: Path) -> None:
        """Should skip normalization when normalize=False."""
        loader = ImageDataLoader()
        data, _ = asyncio.run(loader.load(str(npz_dataset), normalize=False))
        x, _ = data
        assert x.max() > 1.0  # raw pixel values

    def test_load_npz_max_samples(self, npz_dataset: Path) -> None:
        """Should limit samples."""
        loader = ImageDataLoader()
        data, info = asyncio.run(loader.load(str(npz_dataset), max_samples=50))
        assert info.num_samples == 50

    def test_load_npz_class_names(self, npz_dataset: Path) -> None:
        """Should use provided class names."""
        names = ["class_0", "class_1"]
        loader = ImageDataLoader()
        _, info = asyncio.run(loader.load(str(npz_dataset), class_names=names))
        assert info.class_labels == names

    def test_split(self, npz_dataset: Path) -> None:
        """Should split with stratification."""
        loader = ImageDataLoader()
        data, _ = asyncio.run(loader.load(str(npz_dataset)))
        train, test = asyncio.run(loader.split(data, test_size=0.2))

        x_train, _ = train
        x_test, _ = test
        assert len(x_train) == 160
        assert len(x_test) == 40

    def test_file_not_found(self) -> None:
        """Should raise FileNotFoundError."""
        loader = ImageDataLoader()
        with pytest.raises(FileNotFoundError):
            asyncio.run(loader.load("/nonexistent/file.npz"))

    def test_implements_interface(self) -> None:
        """Should be a proper implementation of IDataLoader."""
        assert issubclass(ImageDataLoader, IDataLoader)


# ─── DataLoaderRegistry Tests ─────────────────────────────────────


class TestDataLoaderRegistry:
    """Tests for the plugin registry."""

    def test_register_and_get(self) -> None:
        """Should register and retrieve a DataLoader class."""
        DataLoaderRegistry.register("my-model", TabularDataLoader)
        result = DataLoaderRegistry.get("my-model")
        assert result is TabularDataLoader

    def test_get_unregistered(self) -> None:
        """Should return None for unregistered model."""
        assert DataLoaderRegistry.get("unknown") is None

    def test_clear(self) -> None:
        """Should clear all registrations."""
        DataLoaderRegistry.register("test", TabularDataLoader)
        DataLoaderRegistry.clear()
        assert DataLoaderRegistry.get("test") is None


# ─── resolve_data_loader Tests ─────────────────────────────────────


class TestResolveDataLoader:
    """Tests for the unified resolution function."""

    def test_resolve_from_registry(self) -> None:
        """Should prefer programmatic registry."""
        DataLoaderRegistry.register("my-model", ImageDataLoader)
        loader = resolve_data_loader("my-model")
        assert isinstance(loader, ImageDataLoader)

    def test_resolve_fallback_tabular(self) -> None:
        """Should fall back to TabularDataLoader for unknown models."""
        with patch(
            "phoenix_ml.infrastructure.data_loaders.registry._load_config_fields",
            return_value=("classification", ""),
        ):
            loader = resolve_data_loader("unknown-model")
            assert isinstance(loader, TabularDataLoader)

    def test_resolve_by_task_type_regression(self) -> None:
        """Should resolve TabularDataLoader for regression tasks."""
        with patch(
            "phoenix_ml.infrastructure.data_loaders.registry._load_config_fields",
            return_value=("regression", ""),
        ):
            loader = resolve_data_loader("any-regression")
            assert isinstance(loader, TabularDataLoader)

    def test_resolve_by_task_type_image(self) -> None:
        """Should resolve ImageDataLoader for image_classification tasks."""
        with patch(
            "phoenix_ml.infrastructure.data_loaders.registry._load_config_fields",
            return_value=("image_classification", ""),
        ):
            loader = resolve_data_loader("any-image")
            assert isinstance(loader, ImageDataLoader)
