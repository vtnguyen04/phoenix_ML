# System Design Overview: Phoenix ML Platform

*Phân tích sâu các quyết định kiến trúc đằng sau hệ thống ML inference tự phục hồi.*

## Bài toán

Xây dựng production ML platform cần giải quyết:

1. **Model Serving**: Serve nhiều models đồng thời, latency < 10ms
2. **Model Management**: Champion/Challenger, versioning, A/B testing
3. **Data Drift Detection**: Tự động phát hiện khi production data thay đổi
4. **Self-Healing**: Alert → rollback → retrain tự động
5. **Observability**: Metrics, tracing, dashboards
6. **Scalability**: Horizontal scaling, batch processing

## Quyết định thiết kế chính

### 1. Domain-Driven Design vs Monolithic

**Vấn đề**: ML platform có nhiều concerns khác nhau (inference, monitoring, training, features) mà monolithic sẽ trộn lẫn.

**Giải pháp**: 5 Bounded Contexts, mỗi context có entities, services, repositories riêng.

```
Inference → Model, Prediction, InferenceEngine, RoutingStrategy
Monitoring → DriftReport, DriftCalculator, AlertManager
Training → TrainingJob, TrainingService
Feature Store → FeatureRegistry, FeatureStore
Model Registry → ModelRepository, ArtifactStorage
```

**Kết quả**: Domain logic testable 100% without infrastructure. Thay đổi database = thay 1 file, không ảnh hưởng domain.

### 2. CQRS — Tách Read và Write

**Vấn đề**: Prediction (write-heavy) và model queries (read-heavy) có access patterns khác nhau.

**Giải pháp**: 
- **Commands**: PredictCommand → PredictHandler (write prediction log + publish events)
- **Queries**: GetModelQuery → GetModelQueryHandler (read from DB)

**Benefit**: Commands có thể scale independently, queries có thể cache.

### 3. Event-Driven Architecture — Kafka + EventBus

**Vấn đề**: Prediction logging, metrics publishing, drift detection KHÔNG nên block inference path.

**Giải pháp**: 2-layer event system:
- **DomainEventBus** (in-process): publish Prometheus metrics, update counters → zero latency overhead
- **Kafka** (cross-process): persist events, enable replaying, scale consumers → eventual consistency OK

```
PredictHandler
    ├─ EventBus.publish(PredictionCompleted) → Prometheus metrics (sync)
    └─ KafkaProducer.publish(event) → prediction logging (async)
```

### 4. Strategy Pattern — Traffic Routing

**Vấn đề**: Cần A/B test champion vs challenger models an toàn.

**Giải pháp**: RoutingStrategy abstraction với 4 implementations:

| Strategy | Use case | Risk |
|----------|----------|------|
| SingleModel | Production stable | None |
| ABTest | Compare 2 models | Medium (50/50 split) |
| Canary | Gradual rollout (5%) | Low |
| Shadow | Mirror traffic, compare offline | None (champion always returned) |

### 5. Circuit Breaker — Fault Tolerance

**Vấn đề**: Model engine crash → cascading failures.

**Giải pháp**: 3-state Circuit Breaker:
- **CLOSED** (normal): track failures, open if error rate > threshold
- **OPEN** (blocked): reject all requests, wait timeout
- **HALF_OPEN** (testing): allow limited requests, close if successful

### 6. Plugin Architecture — Model-Agnostic

**Vấn đề**: Mỗi model type cần pre/post processing khác nhau (classification labels, regression scaling, image normalization).

**Giải pháp**: PluginRegistry — register per-model processors:
```python
plugin_registry.register_model(
    model_id="credit-risk",
    preprocessor=PassthroughPreprocessor(),
    postprocessor=ClassificationPostprocessor(labels=["low", "high"]),
    data_loader=TabularDataLoader(),
)
```

**Benefit**: Thêm model mới = thêm 1 YAML config + 1 training script. Zero code changes to core.

## Performance Architecture

### Latency Optimization

| Technique | Implementation | Impact |
|-----------|---------------|--------|
| ONNX Runtime | `onnx_engine.py` | 2-5x faster than sklearn predict |
| Model caching | LRU cache in engine | Load once, predict many |
| Async I/O | `asyncio.to_thread()` for CPU inference | Non-blocking API |
| Connection pooling | SQLAlchemy async pool | Reduce DB connection overhead |

### Throughput Optimization

| Technique | Implementation | Impact |
|-----------|---------------|--------|
| Dynamic batching | `BatchManager` | Collect N requests → 1 batch call |
| Kafka async publishing | Background task | Don't wait for publish ACK |
| gRPC | Binary protocol | 3-5x better than REST for high RPS |

## Tradeoffs

| Decision | Benefit | Cost |
|----------|---------|------|
| DDD layers | Clean, testable, swappable | More files, learning curve |
| ONNX standardization | Single runtime, cross-platform | Conversion step required |
| Kafka | Async, durable, scalable | Extra container, eventual consistency |
| PostgreSQL | ACID, complex queries | Slower than NoSQL for simple KV |
| Redis feature store | Sub-ms feature retrieval | Memory cost, data consistency |

---
*Published: March 2026 · Author: Võ Thành Nguyễn*
