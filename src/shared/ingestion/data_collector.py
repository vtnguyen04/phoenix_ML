import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)


class IDataCollector(ABC):
    """Interface for collecting real-world datasets."""

    @abstractmethod
    async def collect(self) -> pd.DataFrame:
        """Fetches data from a source (Web, API, or Open Datasets)."""
        pass


class CreditDataCollector(IDataCollector):
    """Collects the German Credit Dataset from OpenML."""

    async def collect(self) -> pd.DataFrame:
        logger.info("Fetching Credit Risk dataset from OpenML...")
        data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        df = data.frame

        processed_df = pd.DataFrame()
        processed_df["income"] = df["credit_amount"].astype(float)
        processed_df["debt"] = df["duration"].astype(float)
        processed_df["age"] = df["age"].astype(float)
        processed_df["credit_history"] = df["existing_credits"].astype(float)
        processed_df["target"] = (df["class"] == "good").astype(int)

        return processed_df
