import pytest

from src.domain.inference.services.inference_service import PredictionRequest
from src.domain.inference.services.request_pipeline import (
    CacheHandler,
    RateLimitHandler,
    ValidationHandler,
    build_pipeline,
)


@pytest.mark.asyncio
async def test_validation_passes_valid_request() -> None:
    handler = ValidationHandler()
    request = PredictionRequest(model_id="m1", features=[1.0, 2.0])
    result = await handler.handle(request)
    assert result.model_id == "m1"


@pytest.mark.asyncio
async def test_validation_rejects_missing_model_id() -> None:
    handler = ValidationHandler()
    request = PredictionRequest(model_id="", features=[1.0])
    with pytest.raises(ValueError, match="model_id is required"):
        await handler.handle(request)


@pytest.mark.asyncio
async def test_validation_rejects_no_features_no_entity() -> None:
    handler = ValidationHandler()
    request = PredictionRequest(model_id="m1")
    with pytest.raises(ValueError, match="Either features or entity_id"):
        await handler.handle(request)


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit() -> None:
    handler = RateLimitHandler(max_requests=3)
    request = PredictionRequest(model_id="m1", entity_id="user1", features=[1.0])

    for _ in range(3):
        await handler.handle(request)


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit() -> None:
    handler = RateLimitHandler(max_requests=2)
    request = PredictionRequest(model_id="m1", entity_id="user1", features=[1.0])

    await handler.handle(request)
    await handler.handle(request)

    with pytest.raises(PermissionError, match="Rate limit exceeded"):
        await handler.handle(request)


@pytest.mark.asyncio
async def test_cache_returns_cached_result() -> None:
    handler = CacheHandler()
    request = PredictionRequest(model_id="m1", model_version="v1", features=[1.0])

    result1 = await handler.handle(request)
    result2 = await handler.handle(request)
    assert result1 == result2


@pytest.mark.asyncio
async def test_full_pipeline_chain() -> None:
    pipeline = build_pipeline(max_requests=10)
    request = PredictionRequest(model_id="m1", features=[1.0, 2.0])

    result = await pipeline.handle(request)
    assert result.model_id == "m1"


@pytest.mark.asyncio
async def test_pipeline_rejects_invalid_request() -> None:
    pipeline = build_pipeline()
    request = PredictionRequest(model_id="")

    with pytest.raises(ValueError):
        await pipeline.handle(request)
