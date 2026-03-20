"""Tests for domain training DatasetInfo and IDataLoader interface."""

import pytest

from src.domain.training.services.data_loader_plugin import DatasetInfo, IDataLoader


class TestDatasetInfo:
    def test_defaults(self) -> None:
        info = DatasetInfo()
        assert info.num_samples == 0
        assert info.num_features == 0
        assert info.feature_names == []
        assert info.class_labels == []
        assert info.data_format == "tabular"
        assert info.metadata == {}

    def test_custom_values(self) -> None:
        info = DatasetInfo(
            num_samples=1000,
            num_features=30,
            feature_names=["f1", "f2"],
            class_labels=["good", "bad"],
            data_format="images",
            metadata={"source": "openml"},
        )
        assert info.num_samples == 1000
        assert info.data_format == "images"
        assert info.metadata["source"] == "openml"


class TestIDataLoaderInterface:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            IDataLoader()  # type: ignore[abstract]
