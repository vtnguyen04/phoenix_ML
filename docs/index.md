# Phoenix ML Platform Documentation

Technical documentation for the **Phoenix ML Platform** — a self-healing, model-agnostic real-time ML inference system built with DDD and Clean Architecture.

## Architecture

- **[System Design](architecture/system-design.md)**: Full architecture, Mermaid diagrams, layer structure, design patterns, self-healing flow
- **[DDD Overview](architecture/ddd-overview.md)**: Bounded contexts, entities, value objects, domain services, repository interfaces, CQRS
- **[Frontend Architecture](frontend/architecture.md)**: React + TypeScript dashboard, component breakdown, API integration

## API & Deployment

- **[API Reference](api/reference.md)**: REST endpoints for inference, batch prediction, model management, monitoring, features
- **[Deployment Guide](deployment/docker-stack.md)**: Docker Compose stack (14+ services), DVC pipeline, env config

## Guides

- **[Library API Guide](guides/library-api.md)**: Code examples for using Phoenix ML as a Python library
- **[Contributing](../CONTRIBUTING.md)**: Development setup, code standards, how to add examples, PR checklist

## Architecture Decision Records

- [ADR 001: DDD Architecture](adr/001-use-ddd-architecture.md)
- [ADR 002: ONNX Runtime Standardization](adr/002-use-onnx-runtime.md)
- [ADR 003: Kafka for Event Streaming](adr/003-use-kafka-for-event-streaming.md)
- [ADR 004: Prometheus + Grafana Observability](adr/004-observability-with-prometheus-grafana.md)
- [ADR 005: DVC for Data/Model Versioning](adr/005-dvc-data-versioning.md)

## Blog Posts

- [System Design Overview](blog/system-design-overview.md)
- [Self-Healing ML](blog/self-healing-ml.md)
- [Performance Optimization](blog/performance-optimization.md)

---
*Repository: [phoenix_ML](https://github.com/vtnguyen04/phoenix_ML)*