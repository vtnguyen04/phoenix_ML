import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.value_objects.confidence_score import ConfidenceScore
from src.domain.inference.value_objects.feature_vector import FeatureVector


class ONNXInferenceEngine(InferenceEngine):
    """
    Production implementation of InferenceEngine using ONNX Runtime.
    Supports high-performance execution on CPU and GPU (if configured).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path("/tmp/phoenix/model_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ort.InferenceSession] = {}

    async def load(self, model: Model) -> None:
        """
        Loads an ONNX model into memory.
        """
        if model.unique_key in self._sessions:
            return

        model_path = self._cache_dir / model.id / model.version / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        # Load session in background thread
        session = await asyncio.to_thread(
            ort.InferenceSession, str(model_path), providers=["CPUExecutionProvider"]
        )
        self._sessions[model.unique_key] = session

    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        """
        Executes inference using ONNX Runtime in a background thread.
        """
        results = await self.batch_predict(model, [features])
        return results[0]

    async def batch_predict(
        self, model: Model, features_list: list[FeatureVector]
    ) -> list[Prediction]:
        """
        Executes batch inference using ONNX Runtime.
        """
        if model.unique_key not in self._sessions:
            await self.load(model)

        session = self._sessions[model.unique_key]

        # Prepare input batch
        input_name = session.get_inputs()[0].name

        # Stack all feature vectors into a single batch [N, D]
        batch_data = np.stack([f.values for f in features_list])

        start_time = time.time()

        # Run inference in thread
        outputs = await asyncio.to_thread(session.run, None, {input_name: batch_data})

        latency_ms_total = (time.time() - start_time) * 1000
        avg_latency_ms = latency_ms_total / len(features_list)

        # Process outputs
        result_tensors = outputs[0]
        predictions = []

        for i in range(len(features_list)):
            result: Any
            confidence_val: float = 1.0

            if len(outputs) > 1 and isinstance(outputs[1], list):
                # Case: Sklearn ONNX
                probs_map = outputs[1][i]
                result = result_tensors[i]

                if hasattr(result, "item"):
                    result = result.item()

                lookup_key = result
                if isinstance(result, (np.int64, np.integer)):
                    lookup_key = int(result)

                confidence_val = float(probs_map.get(lookup_key, 1.0))

            elif isinstance(result_tensors, np.ndarray) and result_tensors.ndim > 1:
                row = result_tensors[i]
                if result_tensors.shape[1] == 1:
                    # Case: Regression [N, 1] — return raw value
                    result = float(row[0])
                else:
                    # Case: Multi-class classification [N, C]
                    result = int(np.argmax(row))
                    confidence_val = float(np.max(row))
            else:
                # Fallback
                val = result_tensors[i]
                result = val.tolist() if hasattr(val, "tolist") else val

            # Clamp confidence to [0, 1] for regression models
            # whose raw outputs may exceed the probability range
            clamped_confidence = max(0.0, min(1.0, confidence_val))

            predictions.append(
                Prediction(
                    model_id=model.id,
                    model_version=model.version,
                    result=result,
                    confidence=ConfidenceScore(value=clamped_confidence),
                    latency_ms=avg_latency_ms,
                )
            )

        return predictions

    async def optimize(self, model: Model) -> None:
        """
        Apply ONNX-specific optimizations like quantization if needed.
        """
        pass
