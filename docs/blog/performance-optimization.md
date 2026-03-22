# Performance Optimization: Latency, Throughput, and Batching

*Engineering sub-millisecond predictions at scale với dynamic batching, caching, và ONNX Runtime.*

## Performance Goals

| Metric | Target | Achieved |
|--------|--------|----------|
| P50 Latency | < 5ms | ~2ms |
| P99 Latency | < 20ms | ~8ms |
| Throughput | > 500 RPS | 800+ RPS |
| Model load time | < 1s | ~200ms |

## Optimization Techniques

### 1. ONNX Runtime — Inference Engine

**Why ONNX Runtime?**

```
sklearn.predict():        ~5ms per call (Python GIL overhead)
onnxruntime.run():        ~1ms per call (C++ backend, optimized graph)
```

**Implementation** (`onnx_engine.py`):

```python
class ONNXInferenceEngine:
    async def predict(self, model, features):
        # Run inference on thread pool to avoid blocking event loop
        result = await asyncio.to_thread(
            self._session.run, None, {"input": features.values.reshape(1, -1)}
        )
        return Prediction(result=result[0], ...)
```

Key optimizations:
- **asyncio.to_thread()**: CPU-bound inference on thread pool → API event loop stays responsive
- **Session caching**: Load ONNX session once per model, reuse across requests
- **Graph optimization**: ONNX Runtime auto-applies: constant folding, operator fusion, memory planning

### 2. Dynamic Batching

**Problem**: Individual predictions waste GPU/CPU cycles (kernel launch overhead).

**Solution**: `BatchManager` — automatically collect concurrent requests into batches:

```python
class BatchManager:
    """Collects requests for max_wait_time_ms, then runs batch_predict once."""
    
    async def submit(self, model, features):
        # Add to pending queue
        self._queue.append(features)
        
        # Wait for batch to fill or timeout
        if len(self._queue) >= max_batch_size or elapsed >= max_wait_time_ms:
            # Run single batch_predict call
            results = await engine.batch_predict(model, self._queue)
            # Distribute results back to waiting callers
```

**Config**:
```bash
BATCH_MAX_SIZE=32        # Max items per batch
BATCH_MAX_WAIT_MS=10     # Max wait time before flushing
```

**Impact**:
- 1 request: ~2ms (no batching overhead)
- 32 concurrent requests: ~5ms total (vs 64ms sequential)
- Throughput: 3-5x improvement under load

### 3. Feature Store Caching (Redis)

**Problem**: Feature retrieval can be slow with database lookups.

**Solution**: Redis HMGET — sub-millisecond feature retrieval:

```python
class RedisFeatureStore:
    async def get_online_features(self, entity_id, feature_names):
        key = f"features:{entity_id}"
        values = await self._redis.hmget(key, *feature_names)
        return [float(v) for v in values]
```

**Performance**:
- Redis HMGET: ~0.1ms (10,000x faster than PostgreSQL JOIN)
- Pipeline support: batch feature lookups

### 4. Async I/O Architecture

**Problem**: Synchronous I/O blocks → latency compounds.

**Solution**: Full async stack:

```python
# AsyncSession → PostgreSQL
async with AsyncSession(engine) as session:
    result = await session.execute(query)

# aioredis → Redis
await redis.hmget(key, *fields)

# AIOKafkaProducer → Kafka
await producer.send(topic, message)

# asyncio.to_thread → CPU inference
result = await asyncio.to_thread(session.run, ...)
```

### 5. Connection Pooling

```python
# SQLAlchemy async engine with pool
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
```

### 6. gRPC for High-Throughput Clients

REST vs gRPC:

| Metric | REST (HTTP/JSON) | gRPC (HTTP/2/Protobuf) |
|--------|-----------------|----------------------|
| Serialization | JSON (text) | Protobuf (binary) |
| Payload size | ~200 bytes | ~80 bytes |
| Connection | HTTP/1.1 (per-request) | HTTP/2 (multiplexed) |
| Latency | ~3ms | ~1.5ms |
| Suitable for | Web clients, debugging | Internal services, high RPS |

### 7. Background Task Offloading

Heavy operations offloaded from the critical inference path:

```python
@app.post("/predict")
async def predict(request, background_tasks: BackgroundTasks):
    # Critical path: predict (2ms)
    prediction = await handler.handle(command)
    
    # Background (not blocking response):
    background_tasks.add_task(log_to_postgres, prediction)
    background_tasks.add_task(publish_to_kafka, prediction)
    
    return prediction  # Response sent immediately
```

## Architecture Impact on Latency

```
Request arrives →        0.0ms
Parse + validate →       0.1ms
Feature retrieval →      0.2ms  (Redis HMGET)
Model inference →        1.5ms  (ONNX Runtime)
Response serialize →     0.1ms
───────────────────────────────
Total (P50):             ~2.0ms

Background (async):
  Log to PostgreSQL →    5ms
  Publish to Kafka →     3ms
  Prometheus metrics →   0.1ms
```

## Engine Comparison

| Engine | Best For | P50 Latency | GPU Support |
|--------|----------|-------------|-------------|
| `ONNXInferenceEngine` | General purpose, CPU | ~2ms | Via CUDAExecutionProvider |
| `TensorRTExecutor` | GPU-heavy workloads | ~0.5ms | TensorRT FP16 |
| `TritonInferenceClient` | Distributed serving | ~3ms (network) | Via Triton server |
| `MockInferenceEngine` | Testing | ~0.01ms | N/A |

## Benchmarking

```bash
# Run latency benchmark
uv run python benchmarks/latency_benchmark.py

# Run throughput benchmark
uv run python benchmarks/throughput_benchmark.py

# Load test with Locust
uv run locust -f benchmarks/locustfile.py --host http://localhost:8001

# Memory profiling
uv run python benchmarks/memory_benchmark.py
```

## Scaling Strategy

### Vertical
- Increase `BATCH_MAX_SIZE` (better GPU utilization)
- Use TensorRT engine (2-5x faster on GPU)
- Increase connection pool size

### Horizontal
- Multiple API replicas behind load balancer
- Kafka consumer groups for parallel event processing
- Kubernetes HPA (auto-scale based on CPU/latency)

```yaml
# deploy/helm/phoenix-ml/templates/hpa.yaml
spec:
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        targetAverageUtilization: 80
```

---
*Published: March 2026 · Author: Võ Thành Nguyễn*
