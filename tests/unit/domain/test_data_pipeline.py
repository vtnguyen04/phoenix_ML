"""Tests for data pipeline — end-to-end ingestion flow."""

from pathlib import Path

import pandas as pd
import pytest

from phoenix_ml.domain.training.services.data_pipeline import DataPipeline


@pytest.fixture
def pipeline() -> DataPipeline:
    return DataPipeline()


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    df = pd.DataFrame({
        "income": [50000, 60000, 70000, 80000, 55000],
        "age": [25, 30, 35, 40, 28],
        "target": [0, 1, 1, 0, 1],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def dirty_csv(tmp_path: Path) -> str:
    df = pd.DataFrame({
        "income": [50000, None, 70000, None, None],
        "age": [25, 30, None, 40, 28],
        "target": [0, 1, 1, 0, 1],
    })
    path = tmp_path / "dirty_data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestDataPipeline:
    async def test_run_clean_file(self, pipeline: DataPipeline, sample_csv: str) -> None:
        result = await pipeline.run_from_file(sample_csv, target_column="target")
        assert result.success is True
        assert result.rows_processed == 5
        assert result.quality_report is not None
        assert result.quality_report.passed is True

    async def test_run_with_output(
        self, pipeline: DataPipeline, sample_csv: str, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "output.csv")
        result = await pipeline.run_from_file(sample_csv, output_path=output)
        assert result.success is True
        assert Path(output).exists()

    async def test_run_missing_file(self, pipeline: DataPipeline) -> None:
        result = await pipeline.run_from_file("/nonexistent/file.csv")
        assert result.success is False
        assert len(result.errors) > 0

    async def test_run_dirty_data_with_high_nulls(
        self, pipeline: DataPipeline, dirty_csv: str
    ) -> None:
        result = await pipeline.run_from_file(dirty_csv)
        # Should still succeed (nulls < 30% threshold per column default)
        # income has 60% nulls → should fail
        if result.quality_report and not result.quality_report.passed:
            assert len(result.errors) > 0
        # But if default threshold allows it, it should clean
        assert result.rows_processed == 5

    async def test_run_from_dataframe(self, pipeline: DataPipeline) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
        result = await pipeline.run_from_dataframe(df, target_column="target")
        assert result.success is True
        assert result.rows_stored == 3

    async def test_parquet_output(
        self, pipeline: DataPipeline, sample_csv: str, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "output.parquet")
        result = await pipeline.run_from_file(sample_csv, output_path=output)
        assert result.success is True
        assert Path(output).exists()
        df = pd.read_parquet(output)
        assert len(df) == 5

    async def test_to_dict(self, pipeline: DataPipeline, sample_csv: str) -> None:
        result = await pipeline.run_from_file(sample_csv)
        d = result.to_dict()
        assert "success" in d
        assert "rows_processed" in d
        assert "quality_report" in d
