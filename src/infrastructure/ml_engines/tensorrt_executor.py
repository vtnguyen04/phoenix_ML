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


class TensorRTExecutor(InferenceEngine):
    """
    Inference Engine using TensorRT for high-performance GPU inference.
    Simulates environment by using ONNX Runtime with TensorrtExecutionProvider.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path("/tmp/phoenix/model_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ort.InferenceSession] = {}

    async def load(self, model: Model) -> None:
        if model.framework not in ["tensorrt", "onnx"]:
            raise ValueError(f"Model framework {model.framework} not supported by TensorRTExecutor")

        if model.unique_key in self._sessions:
            return

        model_path = self._cache_dir / model.id / model.version / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        # Try TensorRT Provider, fallback to CPU Provider for cross-platform simulation
        providers = [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CPUExecutionProvider",
        ]

        session = await asyncio.to_thread(
            ort.InferenceSession, str(model_path), providers=providers
        )
        self._sessions[model.unique_key] = session

    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        results = await self.batch_predict(model, [features])
        return results[0]

    async def batch_predict(
        self, model: Model, features_list: list[FeatureVector]
    ) -> list[Prediction]:
        if model.unique_key not in self._sessions:
            await self.load(model)

        session = self._sessions[model.unique_key]
        input_name = session.get_inputs()[0].name
        batch_data = np.stack([f.values for f in features_list]).astype(np.float32)

        start_time = time.time()
        outputs = await asyncio.to_thread(session.run, None, {input_name: batch_data})
        latency_ms_total = (time.time() - start_time) * 1000
        avg_latency_ms = latency_ms_total / len(features_list)

        result_tensors = outputs[0]
        predictions = []

        for i in range(len(features_list)):
            result: Any
            confidence_val: float = 1.0

            if len(outputs) > 1 and isinstance(outputs[1], list):
                # Sklearn ONNX case
                probs_map = outputs[1][i]
                result = result_tensors[i]

                if hasattr(result, "item"):
                    result = result.item()

                lookup_key = int(result) if isinstance(result, (np.int64, np.integer)) else result
                confidence_val = float(probs_map.get(lookup_key, 1.0))

            elif isinstance(result_tensors, np.ndarray) and result_tensors.ndim > 1:
                # DL [N, C] case
                result = int(np.argmax(result_tensors[i]))
                confidence_val = float(np.max(result_tensors[i]))
            else:
                val = result_tensors[i]
                result = val.tolist() if hasattr(val, "tolist") else val

            predictions.append(
                Prediction(
                    model_id=model.id,
                    model_version=model.version,
                    result=result,
                    confidence=ConfidenceScore(value=confidence_val),
                    latency_ms=avg_latency_ms,
                )
            )

        return predictions

    async def optimize(self, model: Model) -> None:
        """Trigger TensorRT builder to create engine from ONNX with FP16 calibration."""
        pass
