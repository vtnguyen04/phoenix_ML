from pathlib import Path

import pytest

from phoenix_ml.infrastructure.feature_store.parquet_feature_store import ParquetFeatureStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> ParquetFeatureStore:
    return ParquetFeatureStore(base_dir=tmp_path / "offline_features")


@pytest.mark.asyncio
async def test_get_historical_features(tmp_store: ParquetFeatureStore) -> None:
    entity_ids = ["user-1", "user-2"]
    feature_names = ["age", "income"]

    results = await tmp_store.get_historical_features(entity_ids, feature_names)

    assert len(results) == 2
    assert results[0]["entity_id"] == "user-1"
    assert "timestamp" in results[0]
    assert results[0]["age"] == 0.5
    assert results[0]["income"] == 0.5


@pytest.mark.asyncio
async def test_extract_training_dataset(tmp_store: ParquetFeatureStore, tmp_path: Path) -> None:
    output_path = tmp_path / "export" / "dataset.parquet"
    query = "SELECT * FROM features WHERE age > 0.5"

    await tmp_store.extract_training_dataset(query, str(output_path))

    assert output_path.exists()
