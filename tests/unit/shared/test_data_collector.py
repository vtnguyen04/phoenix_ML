"""Tests for CreditDataCollector."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from phoenix_ml.shared.ingestion.data_collector import CreditDataCollector, IDataCollector


class TestIDataCollector:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            IDataCollector()  # type: ignore[abstract]


class TestCreditDataCollector:
    async def test_collect_returns_dataframe(self) -> None:
        # Mock fetch_openml to avoid network calls
        mock_df = pd.DataFrame(
            {
                "credit_amount": [1000.0, 2000.0],
                "duration": [12.0, 24.0],
                "age": [25.0, 30.0],
                "existing_credits": [1.0, 2.0],
                "class": ["good", "bad"],
            }
        )
        mock_data = MagicMock()
        mock_data.frame = mock_df

        with patch(
            "phoenix_ml.shared.ingestion.data_collector.fetch_openml",
            return_value=mock_data,
        ):
            collector = CreditDataCollector()
            result = await collector.collect()

        assert isinstance(result, pd.DataFrame)
        assert "income" in result.columns
        assert "debt" in result.columns
        assert "age" in result.columns
        assert "credit_history" in result.columns
        assert "target" in result.columns
        assert len(result) == 2
        assert result["target"].iloc[0] == 1  # "good" -> 1
        assert result["target"].iloc[1] == 0  # "bad" -> 0
