"""
IPreprocessor / IPostprocessor — Plugin Interfaces for I/O Transformation.

Enables the inference pipeline to handle any input/output format:
  - Classification: feature vector → class label + confidence
  - Object Detection: image tensor → bounding boxes + labels
  - Recommendation: user+item IDs → ranked scores
  - Time Series: sequence → future values
  - NLP: text tokens → embeddings / classifications

Users implement these interfaces to transform raw API requests
into model-compatible inputs and model outputs into API responses.
"""

from abc import ABC, abstractmethod
from typing import Any


class IPreprocessor(ABC):
    """Plugin interface for input preprocessing.

    Transforms raw API request data into model-compatible input format.

    Example implementations:
      - TabularPreprocessor: dict of features → float tensor
      - ImagePreprocessor: base64 image → normalized tensor
      - TextPreprocessor: raw text → tokenized tensor
      - TimeSeriesPreprocessor: timestamp+values → sliding window tensor

    Example::

        class ImagePreprocessor(IPreprocessor):
            async def preprocess(self, raw_input, model_config):
                img = decode_base64(raw_input["image"])
                tensor = resize_and_normalize(img, size=640)
                return tensor.flatten().tolist()
    """

    @abstractmethod
    async def preprocess(
        self, raw_input: dict[str, Any], model_config: dict[str, Any]
    ) -> list[float]:
        """Transform raw input into model-compatible feature vector.

        Args:
            raw_input: Raw data from API request.
            model_config: Model configuration for context (feature names, etc.).

        Returns:
            List of float values ready for model inference.
        """
        ...


class IPostprocessor(ABC):
    """Plugin interface for output postprocessing.

    Transforms raw model output into meaningful API response data.

    Example implementations:
      - ClassificationPostprocessor: [0.8, 0.2] → {"label": "good", "confidence": 0.8}
      - DetectionPostprocessor: raw_output → [{"bbox": [...], "label": "car", "conf": 0.9}]
      - RegressionPostprocessor: [42.5] → {"predicted_value": 42.5}

    Example::

        class DetectionPostprocessor(IPostprocessor):
            async def postprocess(self, model_output, model_config):
                boxes = decode_boxes(model_output)
                return {
                    "detections": [
                        {"bbox": b.coords, "label": b.label, "confidence": b.conf}
                        for b in boxes
                    ]
                }
    """

    @abstractmethod
    async def postprocess(
        self, model_output: list[float], model_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Transform raw model output into API response format.

        Args:
            model_output: Raw output from model inference.
            model_config: Model configuration for context (class labels, etc.).

        Returns:
            Dict with processed results for API response.
        """
        ...


class PassthroughPreprocessor(IPreprocessor):
    """Default preprocessor that passes features through unchanged.

    Used for models that accept raw float vectors (e.g. tabular classification).
    """

    async def preprocess(
        self, raw_input: dict[str, Any], model_config: dict[str, Any]
    ) -> list[float]:
        features = raw_input.get("features", [])
        return [float(f) for f in features]


class ClassificationPostprocessor(IPostprocessor):
    """Default postprocessor for binary/multi-class classification.

    Interprets model output as class probabilities.
    """

    BINARY_THRESHOLD = 0.5

    async def postprocess(
        self, model_output: list[float], model_config: dict[str, Any]
    ) -> dict[str, Any]:
        if len(model_output) == 1:
            # Binary classification: single probability
            prob = model_output[0]
            return {
                "prediction": int(prob >= self.BINARY_THRESHOLD),
                "confidence": max(prob, 1.0 - prob),
            }

        # Multi-class: argmax
        max_idx = max(range(len(model_output)), key=lambda i: model_output[i])
        class_labels = model_config.get("class_labels", [])
        label = class_labels[max_idx] if max_idx < len(class_labels) else str(max_idx)

        return {
            "prediction": max_idx,
            "label": label,
            "confidence": model_output[max_idx],
            "probabilities": model_output,
        }
