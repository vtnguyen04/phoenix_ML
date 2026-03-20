# System Design Overview: Phoenix ML Platform

*A deep dive into the architecture decisions behind a self-healing, model-agnostic ML inference system.*

---

## Introduction

The **Phoenix ML Platform** is a production-grade, model-agnostic machine learning inference system designed with Domain-Driven Design (DDD) and SOLID principles at its core. It powers real-time predictions for **any ONNX-exportable model** — from credit risk scoring to image classification — while autonomously monitoring, detecting drift, and self-healing when model performance degrades.

This post explores the key architectural decisions that make the system resilient, maintainable, and performant.

## Architecture: Domain-Driven Design

We chose DDD to manage the inherent complexity of ML systems, which span inference, monitoring, training, and feature management — each a distinct bounded context.

```
src/
├── domain/            ← Pure business logic (no framework deps)
│   ├── inference/     ← Prediction orchestration
│   ├── monitoring/    ← Drift detection, anomaly detection
│   ├── training/      ← Training job lifecycle + plugins
│   ├── feature_store/ ← Feature management
│   └── model_registry/ ← Model versioning
├── application/       ← Use cases (handlers, commands, queries)
│   ├── commands/      ← PredictCommand, BatchPredictCommand
│   ├── handlers/      ← PredictHandler, BatchPredictHandler
│   └── services/      ← MonitoringService
├── infrastructure/    ← Adapters (FastAPI, gRPC, Postgres, Redis, Kafka)
└── shared/            ← Cross-cutting (exceptions, interfaces, utils)
```

### Why DDD?

1. **Bounded Contexts** keep inference, monitoring, and training concerns isolated — changes to drift detection never risk breaking predictions.
2. **Aggregate Roots** (e.g., `Model`, `DriftReport`, `TrainingJob`) enforce invariants — a training job cannot transition from COMPLETED back to RUNNING.
3. **Domain Events** (e.g., `PredictionMade`, `DriftDetected`, `TrainingCompleted`) enable loose coupling between contexts without shared mutable state.

## Model-Agnostic Design

The platform is designed to serve any ML model, not just a specific use case:

| Model | Framework | Task | Features |
|-------|-----------|------|----------|
| Credit Risk | scikit-learn (GBClassifier) | Binary classification | 30 tabular |
| House Price | scikit-learn (Ridge) | Regression | 8 tabular |
| Fraud Detection | XGBoost | Imbalanced classification | 12 tabular |
| Image Classification | sklearn MLP (256→128) | Multi-class (10 classes) | 784 (28×28) |

Adding a new model requires only:
1. A training script in `examples/<name>/train.py`
2. A YAML config in `model_configs/<name>.yaml`
3. Run training → ONNX model is exported automatically

## SOLID Principles in Practice

### Single Responsibility
Each service has one job: `InferenceService` orchestrates predictions, `DriftCalculator` computes statistical tests, `BatchManager` handles request batching. None of them know about HTTP or gRPC.

### Open/Closed
New routing strategies (Canary, Shadow, A/B) implement the `RoutingStrategy` interface without modifying `InferenceService`. New inference engines (ONNX, TensorRT, Triton) implement `InferenceEngine`. The system gains new behavior by *adding* code, not changing it.

### Dependency Inversion
Domain services depend on abstract interfaces (`ModelRepository`, `FeatureStore`, `ArtifactStorage`, `InferenceEngine`). Infrastructure adapters (`PostgresModelRegistry`, `RedisFeatureStore`, `S3ArtifactStorage`, `ONNXInferenceEngine`) are injected at startup via the `Container`.

```python
# Domain defines the contract
class InferenceEngine(ABC):
    @abstractmethod
    def predict(self, model_id: str, version: str, features: list[float]) -> dict: ...

# Infrastructure provides implementations
class ONNXInferenceEngine(InferenceEngine):
    def predict(self, model_id, version, features):
        session = self._sessions[f"{model_id}:{version}"]
        return session.run(None, {"input": np.array([features], dtype=np.float32)})
```

## Dual Transport: FastAPI + gRPC

The platform exposes both REST (FastAPI) and gRPC interfaces:

- **FastAPI** for human-facing APIs (dashboards, monitoring endpoints, batch predictions)
- **gRPC** for high-throughput, low-latency service-to-service inference calls

Both transports share the same `PredictHandler` → `InferenceService` pipeline, ensuring consistent behavior regardless of protocol.

## Key Infrastructure Patterns

| Pattern | Purpose |
|---------|---------|
| **Circuit Breaker** | Prevents cascading failures during model load errors |
| **Dynamic Batching** | Groups concurrent requests for GPU-efficient batch inference |
| **Feature Store** | Online (Redis) for real-time, offline (Parquet) for batch |
| **Event Sourcing** | Kafka-based event log for audit trail and async processing |
| **Request Pipeline** | Chain of Responsibility for composable request processing |
| **Plugin Registry** | Model-agnostic plugin resolution for trainers and data loaders |

## Deployment: Docker Compose + Kubernetes

### Docker Compose (14+ services)
Production-ready stack with all infrastructure:
- **Core**: API, Frontend, PostgreSQL, Redis, Kafka (KRaft)
- **MLOps**: MLflow, Airflow (webserver + scheduler)
- **Observability**: Prometheus, Grafana, Jaeger
- **Storage**: MinIO (S3-compatible)

### Kubernetes (Helm)
The Helm chart includes:
- **ConfigMap/Secret** separation for config vs. credentials
- **StartupProbe** for slow-loading ONNX models
- **PodDisruptionBudget** ensuring availability during rolling updates
- **HPA** for auto-scaling based on CPU/memory metrics
- **Separate gRPC Service** for protocol-specific load balancing

## Lessons Learned

1. **DDD pays off at scale** — the upfront modeling cost prevents "spaghetti ML" where inference, monitoring, and training logic interleave.
2. **Async-first is essential** — `asyncio` throughout the stack prevents blocking during I/O-heavy operations (model loading, feature retrieval).
3. **Model-agnostic from day one** — designing the platform around interfaces rather than specific models means adding new ML use cases takes minutes, not days.
4. **Test the domain, mock the infrastructure** — pure domain logic can be tested without databases, message queues, or ML frameworks.

---

*Phoenix ML Platform — built for reliability, designed for evolution.*
