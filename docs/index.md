# Phoenix ML Platform Documentation

Welcome to the official technical documentation for the **Phoenix ML Platform**. This platform is a state-of-the-art, high-throughput, low-latency system designed for real-time machine learning inference at scale.

Built with **Domain-Driven Design (DDD)** and **SOLID** principles, Phoenix ML provides a robust foundation for deploying, monitoring, and maintaining machine learning models in production environments.

## Technical Core Pillars

### 1. Architectural Integrity (DDD & SOLID)
The platform is partitioned into strictly decoupled layers. By using the **Command Pattern** in the Application layer and **Dependency Inversion** in the Infrastructure layer, we ensure that the core ML logic (Domain) remains pure, testable, and completely independent of external frameworks like FastAPI or ONNX Runtime.

### 2. Real-time Observability & Feedback Loops
Observability is baked into the core. Every inference request is tracked for latency, throughput, and confidence. Using **Prometheus** and **Grafana**, the system provides sub-second visibility into model health. This real-time telemetry allows for proactive identification of performance regressions.

### 3. Autonomous Self-Healing (ML Monitoring)
Phoenix ML addresses the "silent failure" problem in ML. The system features an integrated background monitoring service that performs statistical tests (e.g., **Kolmogorov-Smirnov**) on production data streams. Upon detecting significant **Data Drift**, the system logs alerts and triggers simulated retraining workflows to maintain model accuracy.

### 4. Enterprise-Grade Model Management
With native support for **ONNX Runtime**, the platform handles models from various frameworks (PyTorch, Scikit-Learn, etc.) with unified high-performance execution. It supports sophisticated rollout strategies including **A/B Testing** and **Canary Deployments** via a dynamic model routing engine.

## Navigation Map

### 🏗️ Architecture & Deep-Dive
*   **[System Design & Data Flow](architecture/system-design.md)**: Detailed technical specifications, Sequence Diagrams, and ER Diagrams.
*   **[Frontend Architecture](frontend/architecture.md)**: Breakdown of the React + TypeScript dashboard and state management strategy.

### 🔌 API & Integration
*   **[API Reference](api/reference.md)**: Exhaustive documentation of REST endpoints, JSON schemas, and error handling.
*   **[Deployment Guide](deployment/docker-stack.md)**: Instructions for orchestrating the full stack (Kafka, Redis, Postgres, Monitoring).

### 📜 Architecture Decision Records (ADR)
These documents explain the *rationale* behind our critical technology choices:
*   [ADR 001: Adoption of DDD Architecture](adr/001-use-ddd-architecture.md)
*   [ADR 002: Standardization on ONNX Runtime](adr/002-use-onnx-runtime.md)
*   [ADR 003: Kafka for Real-time Event Streaming](adr/003-use-kafka-for-event-streaming.md)
*   [ADR 004: Metrics-Driven Observability Strategy](adr/004-observability-with-prometheus-grafana.md)

---
**Author: Võ Thành Nguyễn**  
*Senior ML Platform Engineer*  
*Repository: [phoenix_ML](https://github.com/vtnguyen04/phoenix_ML.git)*