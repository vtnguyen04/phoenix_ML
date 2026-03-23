# System Design Overview: Phoenix ML Platform

*Deep analysis of the architecture decisions behind the self-healing ML inference system.*

## Problem Statement

Building a production ML platform requires solving:

1. **Model Serving**: Serve multiple models concurrently with latency < 10 ms
2. **Model Management**: Champion/Challenger, versioning, A/B testing
3. **Data Drift Detection**: Automatically detect when production data changes
4. **Self-Healing**: Alert → rollback → auto-retrain
5. **Observability**: Metrics, tracing, dashboards
6. **Scalability**: Horizontal scaling, batch processing

## Key Design Decisions

### 1. Layered Architecture vs Monolithic

**Problem**: ML platform has many different concerns (inference, monitoring, training, features) that a monolithic approach would mix.

**Solution**: 5 Bounded Contexts, each with its own entities, services, and repositories.

```
Inference → Model, Prediction, InferenceEngine, RoutingStrategy
Monitoring → DriftReport, DriftCalculator, AlertManager
Training → TrainingJob, TrainingService
Feature Store → FeatureRegistry, FeatureStore
Model Registry → ModelRepository, ArtifactStorage
```

**Result**: Domain logic 100% testable without infrastructure. Changing database = changing 1 file, no domain impact.

### 2. CQRS — Separating Read and Write

**Problem**: Prediction (write-heavy) and model queries (read-heavy) have different access patterns.

**Solution**: 
- **Commands**: PredictCommand → PredictHandler (write prediction log + publish events)
- **Queries**: GetModelQuery → GetModelQueryHandler (read from DB)

**Benefit**: Commands can scale independently, queries can be cached.

### 3. Event-Driven Architecture — Kafka + EventBus

**Problem**: Prediction logging, metrics publishing, drift detection should NOT block the inference path.

**Solution**: 2-layer event system:
- **DomainEventBus** (in-process): publish Prometheus metrics, update counters → zero latency overhead
- **Kafka** (cross-process): persist events, enable replaying, scale consumers → eventual consistency OK

```
PredictHandler
    ├─ EventBus.publish(PredictionCompleted) → Prometheus metrics (sync)
    └─ KafkaProducer.publish(event) → prediction logging (async)
```

### 4. Strategy Pattern — Traffic Routing

**Problem**: Need to A/B test champion vs challenger models safely.

**Solution**: RoutingStrategy abstraction with 4 implementations:

| Strategy | Use case | Risk |
|----------|----------|------|
| SingleModel | Production stable | None |
| ABTest | Compare 2 models | Medium (50/50 split) |
| Canary | Gradual rollout (5%) | Low |
| Shadow | Mirror traffic, compare offline | None (champion always returned) |

### 5. Circuit Breaker — Fault Tolerance

**Problem**: Model engine crash → cascading failures.

**Solution**: 3-state Circuit Breaker:
- **CLOSED** (normal): track failures, open if error rate > threshold
- **OPEN** (blocked): reject all requests, wait timeout
- **HALF_OPEN** (testing): allow limited requests, close if successful

### 6. Plugin Architecture — Model-Agnostic

**Problem**: Each model type needs different pre/post processing (classification labels, regression scaling, image normalization).

**Solution**: PluginRegistry — register per-model processors:
```python
plugin_registry.register_model(
    model_id="credit-risk",
    preprocessor=PassthroughPreprocessor(),
    postprocessor=ClassificationPostprocessor(labels=["low", "high"]),
    data_loader=TabularDataLoader(),
)
```

**Benefit**: Adding a new model = 1 YAML config + 1 training script. Zero code changes to core.

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
| Layered architecture | Clean, testable, swappable | More files, learning curve |
| ONNX standardization | Single runtime, cross-platform | Conversion step required |
| Kafka | Async, durable, scalable | Extra container, eventual consistency |
| PostgreSQL | ACID, complex queries | Slower than NoSQL for simple KV |
| Redis feature store | Sub-ms feature retrieval | Memory cost, data consistency |

---
*Published: March 2026*
