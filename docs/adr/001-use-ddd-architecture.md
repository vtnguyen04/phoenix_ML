# ADR 001: Adoption of Domain-Driven Design (DDD)

## Status
✅ **Accepted** — March 2026

## Context

Dự án MLOps platform cần quản lý nhiều domain phức tạp chồng chéo: inference, monitoring, training, feature store, model registry. Mỗi domain có business logic riêng biệt, yêu cầu riêng, và lifecycle riêng.

### Vấn đề

- Code monolithic nhanh chóng trở nên **unmaintainable** khi features tăng
- Business logic bị trộn lẫn với framework code (FastAPI, SQLAlchemy, Redis)
- Khó test domain logic vì phụ thuộc infrastructure
- Thay đổi database/framework ảnh hưởng toàn bộ codebase

## Decision

Áp dụng **Domain-Driven Design (DDD)** kết hợp **Clean Architecture**:

### Bounded Contexts
- **Inference**: Model, Prediction, InferenceEngine, RoutingStrategy, CircuitBreaker
- **Monitoring**: DriftReport, DriftCalculator, AlertManager, AnomalyDetector, RollbackManager
- **Training**: TrainingJob, TrainingService, IDataLoader, ITrainer
- **Feature Store**: FeatureRegistry, FeatureStore, OfflineFeatureStore
- **Model Registry**: ModelRepository, ArtifactStorage

### Layer Rules
```
Infrastructure → Application → Domain → Shared
     ↑ depends on      ↑              ↑
     (implements ABCs)  (orchestrates)  (pure logic)
```

### Design Patterns Applied
- **Repository Pattern**: Abstract data access (ABCs in domain, implementations in infrastructure)
- **Strategy Pattern**: RoutingStrategy (A/B, Canary, Shadow)
- **Observer Pattern**: DomainEventBus for event-driven decoupling
- **CQRS**: Separate PredictCommand handlers from Query handlers
- **Factory Pattern**: Engine factory in container.py
- **Plugin Pattern**: PluginRegistry for model-specific pre/post processors

## Consequences

### Positive
- ✅ Domain logic testable **without** any framework or infrastructure
- ✅ Infrastructure swap (Redis → Memcached, PostgreSQL → MongoDB) without touching domain
- ✅ Team có thể work on different bounded contexts independently
- ✅ Ubiquitous Language: code names match domain concepts
- ✅ 87% test coverage achieved easily due to testable architecture

### Negative
- ❌ More files và directories (40+ domain files)
- ❌ Steeper learning curve cho developers mới
- ❌ Some boilerplate (ABC definitions, dependency injection setup)

## References

- Eric Evans, "Domain-Driven Design", 2003
- Robert C. Martin, "Clean Architecture", 2017
- Vaughn Vernon, "Implementing Domain-Driven Design", 2013