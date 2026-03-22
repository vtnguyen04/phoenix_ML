from pathlib import Path

import numpy as np
import pytest

from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.inference.value_objects.feature_vector import FeatureVector
from phoenix_ml.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from phoenix_ml.shared.utils.model_generator import generate_simple_onnx


@pytest.mark.asyncio
async def test_real_credit_risk_model_inference() -> None:
    # 1. Setup
    cache_dir = Path("tests/data/model_cache")
    model_path = cache_dir / "credit-risk" / "v1" / "model.onnx"

    # Ensure model file exists for CI environment
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        generate_simple_onnx(model_path, n_features=30)

    engine = ONNXInferenceEngine(cache_dir=cache_dir)

    model = Model(
        id="credit-risk",
        version="v1",
        uri="local://tests/data/model_cache",
        framework="onnx",
    )

    # Case 1: Likely Positive
    input_data = np.array([0.5] * 30, dtype=np.float32)
    features = FeatureVector(values=input_data.tolist())

    # 3. Predict
    prediction = await engine.predict(model, features)

    # 4. Verify
    print(f"Prediction Result: {prediction.result}")
    print(f"Confidence: {prediction.confidence.value}")

    assert prediction.model_id == "credit-risk"
    # Scikit-learn ONNX model returns label as result usually
    # Our engine logic: result = raw_result[0] (or processed)

    # Kiểm tra xem engine có chạy thành công không
    assert prediction.result is not None
    assert isinstance(prediction.latency_ms, float)
    assert prediction.latency_ms > 0

    # Case 2: Likely Negative
    input_data_neg = np.array([-0.5] * 30, dtype=np.float32)
    features_neg = FeatureVector(values=input_data_neg.tolist())
    prediction_neg = await engine.predict(model, features_neg)

    print(f"Prediction Neg: {prediction_neg.result}")

    # Expect different outcomes or at least valid execution
    assert prediction_neg.result is not None
