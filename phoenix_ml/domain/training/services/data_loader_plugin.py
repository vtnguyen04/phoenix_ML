"""
IDataLoader — Plugin Interface for Dataset Loading.

Enables loading any data format: tabular CSV, images with annotations,
text corpora, time series, audio files, etc.

Each ML problem type implements this interface to load its specific
data format into a standardized structure the framework can manage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset.

    Attributes:
        num_samples: Total number of samples.
        num_features: Number of input features (0 for image/audio).
        feature_names: Ordered list of feature names (empty for unstructured).
        class_labels: List of class labels for classification tasks.
        data_format: Description of data format ("tabular", "images", "text", etc.).
        metadata: Additional dataset-specific info.
    """

    num_samples: int = 0
    num_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)
    data_format: str = "tabular"
    metadata: dict[str, Any] = field(default_factory=dict)


class IDataLoader(ABC):
    """Plugin interface for dataset loading.

    Implement this for your specific data format:
      - Tabular: CSV, Parquet, database queries
      - Images: COCO, YOLO format, Pascal VOC
      - Text: JSON-lines, CSV with text columns
      - Time Series: CSV with timestamps, Parquet
      - Audio: WAV files with transcriptions

    Example::

        class COCODataLoader(IDataLoader):
            async def load(self, data_path, **kwargs):
                # Load COCO annotation format for object detection
                return data, DatasetInfo(
                    num_samples=len(images),
                    data_format="coco_detection",
                    class_labels=["person", "car", "dog", ...],
                )
    """

    @abstractmethod
    async def load(self, data_path: str, **kwargs: Any) -> tuple[Any, DatasetInfo]:
        """Load a dataset from the given path.

        Args:
            data_path: Path to the dataset (file, directory, or URI).
            **kwargs: Loader-specific options (e.g., split, max_samples).

        Returns:
            Tuple of (data_object, dataset_info).
            data_object type varies by implementation (DataFrame, dict, etc.).
        """
        ...

    @abstractmethod
    async def split(
        self,
        data: Any,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[Any, Any]:
        """Split data into train and test sets.

        Args:
            data: The data object returned by load().
            test_size: Fraction of data for testing.
            random_seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_data, test_data).
        """
        ...
