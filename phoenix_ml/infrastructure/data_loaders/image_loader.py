# ruff: noqa: PLC0415, PLR2004
"""
ImageDataLoader — Generic NPZ/directory loader for image ML tasks.

Works for classification, object detection, segmentation, or any task
where data is stored as numpy arrays (NPZ) or image directories.

Supported formats:
    - .npz: Compressed numpy (X_train, y_train arrays)
    - Directory: images/ ├── class_0/ ├── class_1/ ...

Usage:
    loader = ImageDataLoader()
    data, info = await loader.load("data/image_class/dataset.npz")
    train, test = await loader.split(data, test_size=0.2)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from phoenix_ml.domain.training.services.data_loader_plugin import DatasetInfo, IDataLoader

logger = logging.getLogger(__name__)


class ImageDataLoader(IDataLoader):
    """Load image datasets from NPZ archives or directory structures.

    Works with any image ML task: classification, detection, segmentation.
    Returns (X, y) numpy arrays with metadata in DatasetInfo.
    """

    async def load(self, data_path: str, **kwargs: Any) -> tuple[Any, DatasetInfo]:
        """Load image dataset from NPZ or directory.

        Args:
            data_path: Path to .npz file or image directory.
            **kwargs:
                x_key: Key for features in NPZ (default: "X").
                y_key: Key for labels in NPZ (default: "y").
                normalize: Whether to normalize pixel values (default: True).
                max_samples: Limit number of samples.
                class_names: List of class label names.

        Returns:
            Tuple of ((X, y) arrays, DatasetInfo metadata).
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        if path.suffix == ".npz":
            return await self._load_npz(path, **kwargs)
        if path.is_dir():
            return await self._load_directory(path, **kwargs)
        raise ValueError(f"Unsupported format: {path.suffix}. Use .npz or directory.")

    async def _load_npz(self, path: Path, **kwargs: Any) -> tuple[Any, DatasetInfo]:
        """Load from compressed numpy archive."""
        x_key = kwargs.get("x_key", "X")
        y_key = kwargs.get("y_key", "y")
        normalize = kwargs.get("normalize", True)
        max_samples = kwargs.get("max_samples")
        class_names = kwargs.get("class_names", [])

        data = np.load(path)
        x_data = data[x_key].astype(np.float32)
        y_data = data[y_key]

        if max_samples and len(x_data) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(x_data))[:max_samples]
            x_data = x_data[indices]
            y_data = y_data[indices]

        if normalize and x_data.max() > 1.0:
            x_data = x_data / 255.0

        n_features = int(np.prod(x_data.shape[1:]))
        n_classes = len(np.unique(y_data))

        # Infer image dimensions
        ndim = len(x_data.shape)
        if ndim == 2:
            side = int(np.sqrt(x_data.shape[1]))
            image_size = f"{side}x{side}"
        elif ndim >= 3:
            image_size = "x".join(str(s) for s in x_data.shape[1:])
        else:
            image_size = "unknown"

        # Flatten for ONNX if needed
        if ndim > 2:
            x_data = x_data.reshape(len(x_data), -1)

        if not class_names:
            class_names = [str(i) for i in range(n_classes)]

        info = DatasetInfo(
            num_samples=len(x_data),
            num_features=n_features,
            feature_names=[f"pixel_{i}" for i in range(min(n_features, 10))],
            class_labels=class_names,
            data_format="image_npz",
            metadata={
                "image_size": image_size,
                "n_classes": n_classes,
                "normalized": normalize,
                "source_path": str(path),
            },
        )

        logger.info(
            "Loaded %d images (%s, %d classes) from %s",
            info.num_samples,
            image_size,
            n_classes,
            path.name,
        )
        return (x_data, y_data), info

    async def _load_directory(self, path: Path, **kwargs: Any) -> tuple[Any, DatasetInfo]:
        """Load from directory structure: path/class_0/, path/class_1/, ..."""
        from PIL import Image

        max_samples = kwargs.get("max_samples")
        normalize = kwargs.get("normalize", True)
        img_size = kwargs.get("img_size", (28, 28))

        class_dirs = sorted([d for d in path.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_dirs]

        images: list[np.ndarray] = []
        labels: list[int] = []

        for label_idx, class_dir in enumerate(class_dirs):
            for img_file in sorted(class_dir.glob("*")):
                if img_file.suffix.lower() in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                ):
                    img = Image.open(img_file).convert("L").resize(img_size)
                    arr = np.array(img, dtype=np.float32).flatten()
                    if normalize:
                        arr = arr / 255.0
                    images.append(arr)
                    labels.append(label_idx)

                if max_samples and len(images) >= max_samples:
                    break
            if max_samples and len(images) >= max_samples:
                break

        x_data = np.array(images, dtype=np.float32)
        y_data = np.array(labels, dtype=int)

        info = DatasetInfo(
            num_samples=len(x_data),
            num_features=int(np.prod(img_size)),
            class_labels=class_names,
            data_format="image_directory",
            metadata={
                "image_size": f"{img_size[0]}x{img_size[1]}",
                "n_classes": len(class_names),
                "source_path": str(path),
            },
        )

        logger.info(
            "Loaded %d images from directory %s (%d classes)",
            len(x_data),
            path.name,
            len(class_names),
        )
        return (x_data, y_data), info

    async def split(
        self,
        data: Any,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[Any, Any]:
        """Split (X, y) into train and test with stratification."""
        from sklearn.model_selection import train_test_split

        x_data, y_data = data

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=test_size,
            random_state=random_seed,
            stratify=y_data,
        )

        logger.info("Split: train=%d, test=%d", len(x_train), len(x_test))
        return (x_train, y_train), (x_test, y_test)
