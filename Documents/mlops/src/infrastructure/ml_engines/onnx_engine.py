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
    Real implementation of InferenceEngine using ONNX Runtime.
    Includes thread-safe execution and session caching.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._lock = asyncio.Lock()

    async def load(self, model: Model) -> None:
        """
        Loads the ONNX model into an InferenceSession if not already loaded.
        """
        async with self._lock:
            if model.unique_key in self._sessions:
                return

            local_path = self._cache_dir / model.id / model.version / "model.onnx"
            if not local_path.exists():
                raise FileNotFoundError(f"Model file not found at {local_path}")

            # Load session in a thread to not block event loop
            session = await asyncio.to_thread(
                ort.InferenceSession, 
                str(local_path),
                providers=['CPUExecutionProvider'] # Default to CPU
            )
            self._sessions[model.unique_key] = session

    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        """
        Executes inference using ONNX Runtime in a background thread.
        """
        if model.unique_key not in self._sessions:
            await self.load(model)

        session = self._sessions[model.unique_key]
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        # Add batch dimension if necessary (e.g., [N] -> [1, N])
        input_data = features.values.reshape(1, -1)

        start_time = time.time()
        
        # Run inference in thread
        outputs = await asyncio.to_thread(
            session.run, 
            None, 
            {input_name: input_data}
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Process output
        # Sklearn-ONNX typically returns: [label_tensor, probability_map_list]
        # Deep Learning ONNX typically returns: [probability_tensor]
        
        result_tensor = outputs[0]
        result: Any
        confidence_val: float = 1.0

        if len(outputs) > 1 and isinstance(outputs[1], list):
            # Case: Sklearn ONNX (Label + Probs)
            # outputs[1] is a list of dicts (one per sample in batch)
            probs_map = outputs[1][0] # Batch size is 1
            
            # Get label
            if isinstance(result_tensor, np.ndarray):
                result = result_tensor[0]
            else:
                result = result_tensor

            # Convert numpy scalar to python native type
            if hasattr(result, "item"):
                result = result.item()

            # Get confidence
            if isinstance(probs_map, dict):
            # result might be numpy scalar, convert to native python type 
            # for key lookup if needed
            # but usually dict keys match the label type
            # Handle numpy int64 key lookup issues
                lookup_key = result
                if isinstance(result, np.int64):
                    lookup_key = int(result)
                    
                confidence_val = float(probs_map.get(lookup_key, 1.0))
                
        elif isinstance(result_tensor, np.ndarray) and result_tensor.ndim > 1:
            # Case: Deep Learning (Logits/Probs) -> [1, C]
            result = int(np.argmax(result_tensor[0]))
            confidence_val = float(np.max(result_tensor[0]))
        else:
            # Fallback
            result = result_tensor.tolist()

        return Prediction(
            model_id=model.id,
            model_version=model.version,
            result=result,
            confidence=ConfidenceScore(value=confidence_val),
            latency_ms=latency_ms
        )

    async def optimize(self, model: Model) -> None:
        """
        Apply ONNX-specific optimizations like quantization if needed.
        """
        pass
