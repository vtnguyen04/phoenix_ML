# ADR 001: Adoption of Domain-Driven Design (DDD)

## Status
Accepted

## Context
Machine Learning systems often suffer from "Glue Code" anti-patterns where business logic (inference rules, drift thresholds) interacts directly with infrastructure (databases, APIs). This high coupling makes the system fragile, untestable, and difficult to adapt to new requirements (e.g., switching from REST to gRPC, or Redis to Cassandra).

## Decision
We adopted **Domain-Driven Design (DDD)** to structure the codebase into four distinct layers:

1.  **Domain Layer** (Core): Contains high-level rules, Entities (Model, Prediction), and Value Objects (ConfidenceScore). This layer has **ZERO dependencies** on external frameworks.
2.  **Application Layer**: Orchestrates use cases using the Command Pattern (e.g., `PredictHandler`, `MonitoringService`). It coordinates the Domain objects but does not contain business rules.
3.  **Infrastructure Layer**: Implements interfaces defined in the Domain (Adapters). Includes FastAPI, Redis, PostgreSQL, and ONNX Runtime configurations.
4.  **Shared Kernel**: Utilities and common types shared across layers.

## Consequences
### Positive
*   **Testability**: Core logic can be unit-tested in isolation without mocking heavy infrastructure.
*   **Flexibility**: Infrastructure components can be swapped (e.g., replacing Redis with Memcached) without touching the Domain logic.
*   **Maintainability**: Clear boundaries prevent "spaghetti code" as the system grows.

### Negative
*   **Complexity**: Increased number of files and boilerplate code compared to a simple script-based approach.
*   **Learning Curve**: Requires team familiarity with DDD concepts (Aggregates, Repositories, Services).