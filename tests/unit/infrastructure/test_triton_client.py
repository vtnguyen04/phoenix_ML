from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from httpx import Response

from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.inference.value_objects.feature_vector import FeatureVector
from phoenix_ml.infrastructure.ml_engines.triton_client import TritonInferenceClient


@pytest.fixture
def mock_model() -> Model:
    return Model(
        id="test-triton-model",
        version="v1",
        uri="local://test",
        framework="onnx",
        metadata={},
    )


@pytest.mark.asyncio
async def test_triton_client_predict_fallback(mock_model: Model) -> None:
    """Test fallback logic when Triton is unreachable."""
    client = TritonInferenceClient(triton_url="http://invalid-url:9999")
    features = FeatureVector(values=np.array([0.1] * 30, dtype=np.float32))

    # Should gracefully catch connection error and return mock/fallback prediction
    prediction = await client.predict(mock_model, features)

    assert prediction.model_id == "test-triton-model"
    assert prediction.result == 1
    assert prediction.confidence.value == 0.95


@pytest.mark.asyncio
async def test_triton_client_predict_success(mock_model: Model) -> None:
    """Test successful API call using mocked httpx.AsyncClient."""
    mock_response = Response(
        200,
        json={"outputs": [{"name": "output", "datatype": "FP32", "shape": [1, 1], "data": [0.8]}]},
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        client = TritonInferenceClient()
        features = FeatureVector(values=np.array([0.1] * 30, dtype=np.float32))
        prediction = await client.predict(mock_model, features)

        assert prediction.result == 1  # 0.8 > 0.5 threshold
        assert prediction.confidence.value == 0.8
        mock_post.assert_awaited_once()

    await client.close()
