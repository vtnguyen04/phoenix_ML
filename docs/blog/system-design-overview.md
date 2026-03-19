# System Design Overview: Phoenix ML Platform

*A deep dive into the architecture decisions behind a self-healing, real-time ML inference system.*

---

## Introduction

The **Phoenix ML Platform** is a production-grade machine learning inference system designed with Domain-Driven Design (DDD) and SOLID principles at its core. It powers real-time predictions for credit risk assessment while autonomously monitoring, detecting drift, and self-healing when model performance degrades.

This post explores the key architectural decisions that make the system resilient, maintainable, and performant.

## Architecture: Domain-Driven Design

We chose DDD to manage the inherent complexity of ML systems, which span inference, monitoring, training, and feature management — each a distinct bounded context.

```
src/
├── domain/            ← Pure business logic (no framework deps)
│   ├── inference/     ← Prediction orchestration
│   ├── monitoring/    ← Drift detection, anomaly detection
│   ├── training/      ← Training job lifecycle
│   ├── feature_store/ ← Feature management
│   └── model_registry/ ← Model versioning
├── application/       ← Use cases (handlers, commands, queries)
├── infrastructure/    ← Adapters (FastAPI, gRPC, Postgres, Redis, Kafka)
└── shared/            ← Cross-cutting concerns
```

### Why DDD?

1. **Bounded Contexts** keep inference, monitoring, and training concerns isolated — changes to drift detection never risk breaking predictions.
2. **Aggregate Roots** (e.g., `TrainingJob`, `Prediction`) enforce invariants — a training job cannot transition from COMPLETED back to RUNNING.
3. **Domain Events** (e.g., `TrainingCompleted`) enable loose coupling between contexts without shared mutable state.

## SOLID Principles in Practice

### Single Responsibility
Each service has one job: `InferenceService` orchestrates predictions, `DriftCalculator` computes statistical tests, `BatchManager` handles request batching. None of them know about HTTP or gRPC.

### Open/Closed
New routing strategies (Canary, Shadow, A/B) implement the `RoutingStrategy` interface without modifying `InferenceService`. The system gains new behavior by *adding* code, not changing it.

### Dependency Inversion
Domain services depend on abstract interfaces (`ModelRepository`, `FeatureStore`, `ArtifactStorage`). Infrastructure adapters (`InMemoryModelRepository`, `RedisFeatureStore`, `S3ArtifactStorage`) are injected at startup.

```python
# Domain defines the contract
class ModelRepository(ABC):
    @abstractmethod
    async def get_champion(self, model_id: str) -> Model | None: ...

# Infrastructure provides the implementation
class InMemoryModelRepository(ModelRepository):
    async def get_champion(self, model_id: str) -> Model | None:
        for m in self._models.values():
            if m.id == model_id and m.metadata.get("role") == "champion":
                return m
        return None
```

## Dual Transport: FastAPI + gRPC

The platform exposes both REST (FastAPI) and gRPC interfaces:

- **FastAPI** for human-facing APIs (dashboards, monitoring endpoints)
- **gRPC** for high-throughput, low-latency service-to-service inference calls

Both transports share the same `PredictHandler` → `InferenceService` pipeline, ensuring consistent behavior regardless of protocol.

## Key Infrastructure Patterns

| Pattern | Purpose |
|---------|---------|
| **Circuit Breaker** | Prevents cascading failures during model load errors |
| **Dynamic Batching** | Groups concurrent requests for GPU-efficient batch inference |
| **Feature Store** | Online (Redis) and offline (Parquet) feature retrieval |
| **Event Sourcing** | Kafka-based event log for audit trail and async processing |

## Deployment: Kubernetes with Helm

The Helm chart includes:
- **ConfigMap/Secret** separation for config vs. credentials
- **StartupProbe** for slow-loading ONNX models
- **PodDisruptionBudget** ensuring availability during rolling updates
- **HPA** for auto-scaling based on CPU/memory metrics
- **Separate gRPC Service** for protocol-specific load balancing

## Lessons Learned

1. **DDD pays off at scale** — the upfront modeling cost prevents "spaghetti ML" where inference, monitoring, and training logic interleave.
2. **Async-first is essential** — `asyncio` throughout the stack prevents blocking during I/O-heavy operations (model loading, feature retrieval).
3. **Test the domain, mock the infrastructure** — 88%+ coverage by testing pure domain logic without needing databases or message queues.

---

*Phoenix ML Platform — built for reliability, designed for evolution.*
