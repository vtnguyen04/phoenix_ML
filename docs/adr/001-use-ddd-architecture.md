# ADR 001: Adoption of Domain-Driven Design (DDD)

## Status
✅ **Accepted** — March 2026

## Context

The MLOps platform project needs to manage multiple complex, overlapping domains: inference, monitoring, training, feature store, model registry. Each domain has its own business logic, requirements, and lifecycle.

### Problem

- Monolithic code quickly becomes **unmaintainable** as features grow
- Business logic is mixed with framework code (FastAPI, SQLAlchemy, Redis)
- Domain logic is hard to test due to infrastructure dependencies
- Changing database/framework affects the entire codebase

## Decision

Adopt **Domain-Driven Design (DDD)** combined with **Clean Architecture**:

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
- ✅ Team can work on different bounded contexts independently
- ✅ Ubiquitous Language: code names match domain concepts
- ✅ 87% test coverage achieved easily due to testable architecture

### Negative
- ❌ More files and directories (40+ domain files)
- ❌ Steeper learning curve for new developers
- ❌ Some boilerplate (ABC definitions, dependency injection setup)

## References

- Eric Evans, "Domain-Driven Design", 2003
- Robert C. Martin, "Clean Architecture", 2017
- Vaughn Vernon, "Implementing Domain-Driven Design", 2013