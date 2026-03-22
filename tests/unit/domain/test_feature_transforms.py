"""Tests for feature transforms and pipeline."""

import pandas as pd
import pytest

from phoenix_ml.domain.feature_store.services.feature_transforms import (
    FeaturePipeline,
    Imputer,
    LogTransform,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "income": [50000.0, 60000.0, 70000.0, 80000.0],
        "age": [25.0, 30.0, 35.0, 40.0],
        "category": ["A", "B", "A", "C"],
    })


class TestStandardScaler:
    def test_zero_mean(self, sample_df: pd.DataFrame) -> None:
        scaler = StandardScaler(columns=["income", "age"])
        result = scaler.fit_transform(sample_df)
        assert abs(result["income"].mean()) < 0.01
        assert abs(result["age"].mean()) < 0.01

    def test_unit_variance(self, sample_df: pd.DataFrame) -> None:
        scaler = StandardScaler(columns=["income"])
        result = scaler.fit_transform(sample_df)
        assert abs(result["income"].std() - 1.0) < 0.1


class TestMinMaxScaler:
    def test_range_0_1(self, sample_df: pd.DataFrame) -> None:
        scaler = MinMaxScaler(columns=["income"])
        result = scaler.fit_transform(sample_df)
        assert result["income"].min() >= -0.01
        assert result["income"].max() <= 1.01

    def test_preserves_order(self, sample_df: pd.DataFrame) -> None:
        scaler = MinMaxScaler(columns=["age"])
        result = scaler.fit_transform(sample_df)
        assert list(result["age"]) == sorted(result["age"])


class TestLogTransform:
    def test_reduces_large_values(self) -> None:
        df = pd.DataFrame({"val": [1, 10, 100, 1000]})
        transform = LogTransform(columns=["val"])
        result = transform.fit_transform(df)
        assert result["val"].max() < 1000

    def test_handles_zero(self) -> None:
        df = pd.DataFrame({"val": [0.0, 1.0, 2.0]})
        transform = LogTransform(columns=["val"])
        result = transform.fit_transform(df)
        assert result["val"].iloc[0] == 0.0  # log1p(0) = 0


class TestImputer:
    def test_median_imputation(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
        imputer = Imputer(strategy="median")
        result = imputer.fit_transform(df)
        assert not result["a"].isnull().any()
        assert result["a"].iloc[2] == 2.0  # median

    def test_mean_imputation(self) -> None:
        df = pd.DataFrame({"a": [2.0, 4.0, None]})
        imputer = Imputer(strategy="mean")
        result = imputer.fit_transform(df)
        assert result["a"].iloc[2] == 3.0

    def test_constant_imputation(self) -> None:
        df = pd.DataFrame({"a": [1.0, None]})
        imputer = Imputer(strategy="constant", fill_value=-1.0)
        result = imputer.fit_transform(df)
        assert result["a"].iloc[1] == -1.0


class TestOneHotEncoder:
    def test_encodes_categories(self, sample_df: pd.DataFrame) -> None:
        encoder = OneHotEncoder(columns=["category"])
        result = encoder.fit_transform(sample_df)
        assert "category" not in result.columns
        assert "category_B" in result.columns or "category_C" in result.columns

    def test_drop_first(self, sample_df: pd.DataFrame) -> None:
        encoder = OneHotEncoder(columns=["category"], drop_first=True)
        result = encoder.fit_transform(sample_df)
        # With drop_first, first category "A" should not have its own column
        assert "category_A" not in result.columns


class TestFeaturePipeline:
    def test_chained_transforms(self, sample_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline(steps=[
            Imputer(strategy="median"),
            StandardScaler(columns=["income", "age"]),
        ])
        result = pipeline.fit_transform(sample_df)
        assert abs(result["income"].mean()) < 0.01

    def test_fit_then_transform(self, sample_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        pipeline.add(MinMaxScaler(columns=["income"]))
        pipeline.fit(sample_df)
        result = pipeline.transform(sample_df)
        assert result["income"].min() >= -0.01

    def test_describe(self) -> None:
        pipeline = FeaturePipeline(steps=[
            Imputer(),
            StandardScaler(),
        ])
        desc = pipeline.describe()
        assert len(desc) == 2
        assert desc[0]["type"] == "Imputer"
        assert desc[1]["type"] == "StandardScaler"
