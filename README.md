# Phoenix ML Platform

> High-Throughput, Low-Latency Real-time ML Inference System with Autonomous Self-Healing

**[🌐 View Live Documentation](https://vtnguyen04.github.io/phoenix_ML/)**

Phoenix ML is an enterprise-grade machine learning inference platform designed for high-availability, scalability, and deep observability. It implements Domain-Driven Design (DDD) to isolate complex ML orchestration from infrastructure concerns, ensuring a robust and maintainable system.

---

## Architectural Philosophy

The system is built upon four fundamental technical pillars:

### 1. Domain-Driven Design (DDD)
The codebase is strictly partitioned into four layers:
-   **Domain Layer**: Contains pure business logic, entities (Model, Prediction), and value objects. It is framework-agnostic and maintains zero external dependencies.
-   **Application Layer**: Orchestrates use cases (Predict, Load Model, Monitor) using the Command Pattern. This layer handles the execution flow without being concerned with data persistence or API details.
-   **Infrastructure Layer**: Provides concrete implementations for the interfaces defined in the Domain. This includes FastAPI for the REST gateway, ONNX Runtime for model execution, and Redis for the online feature store.
-   **Shared Kernel**: Houses common utilities, base exceptions, and shared interfaces used across all layers.

### 2. Event-Driven Observability
Every inference request is treated as an event. These events are published asynchronously to **Apache Kafka**, decoupling the critical inference path from auxiliary tasks like auditing and drift analysis. This ensures that logging and monitoring overhead never impacts client-facing latency.

### 3. Autonomous Self-Healing
The platform features an integrated background monitoring service. It performs real-time statistical analysis (Kolmogorov-Smirnov Test) on production data streams to detect Data Drift. Upon detection of significant distribution shifts, the system automatically updates Prometheus metrics and triggers simulated retraining workflows.

### 4. High-Performance Execution
Standardized on **ONNX Runtime**, the platform provides unified high-performance execution for models trained in any framework (PyTorch, Scikit-Learn, etc.). Combined with a dynamic model routing engine, it supports seamless A/B testing and Canary deployments.

---

## Recent Milestones & Enhancements

- **Frontend SOLID Refactoring**: Migrated the entire React application to highly decoupled, state-isolated components based on SOLID principles. Features include custom React hooks (`useModels`, `useDrift`) and robust unit testing covering isolated UI components.
- **Backend Quality Assurance**: Fully mitigated code smells using `Ruff` and strict type safety enforced by `Mypy`. Removed legacy structural scripts in favor of dynamic Python setups configured by `pyproject.toml`.
- **Comprehensive Benchmarking**: Introduced latency (`latency_benchmark.py`) and throughput (`throughput_benchmark.py`) benchmarks to validate concurrent execution metrics on production hardware.
- **Test-Driven Operations**: Integrated dynamic test mocking via an in-memory lifespan feature seeder that populates `customer-good` mock records during 100% automated End-to-End API test environments.
- **Continuous Integration (CI/CD)**: Added parallel GitHub Actions workflows spanning Pytest integration pipelines and Vitest DOM testing to guarantee flawless quality gates before code merger.

---

## Project Structure

The project follows a modular structure reflecting the Inversion of Control principle:

```text
.
├── src/
│   ├── domain/              # Pure Logic Layer (Zero dependencies)
│   ├── application/         # Orchestration Layer (Commands & Handlers)
│   ├── infrastructure/      # Implementation Layer (Adapters)
│   │   ├── http/            # FastAPI configuration & Lifespan triggers
│   │   ├── ml_engines/      # ONNX and TensorRT implementations
│   │   ├── messaging/       # Kafka producers and consumers
│   │   ├── persistence/     # PostgreSQL, MLflow, and Redis repositories
│   │   └── artifact_storage/# Local / S3 artifact loading
│   └── shared/              # Utilities and Base interfaces
├── frontend/                # React + TypeScript Control Plane
│   ├── src/api/             # Standardized API client
│   ├── src/components/      # Reusable UI components
│   ├── src/hooks/           # Custom state hooks (usePredict, useMetrics)
│   ├── src/types/           # Shared TypeScript interfaces
│   └── src/test/            # 80+ isolated RTL/Vitest suites
├── tests/                   # Backend Comprehensive Test Suite
│   ├── unit/                # Per-layer isolation tests
│   ├── integration/         # DB & Feature Store boundary verification
│   └── e2e/                 # Full prediction & feedback pipelines
├── scripts/                 # MLOps Pipelines (Training, Seeders)
├── benchmarks/              # Performance validation (Latency, Throughput)
├── deploy/                  # Production Helm Charts & Manifests
├── grafana/                 # Monitoring Dashboards (Provisioning)
├── .github/workflows/       # Full-stack CI/CD (Ruff, Mypy, Vitest, Pytest)
└── compose.yaml             # Multi-container stack orchestration
```

---

## Technology Stack

| Category | Component |
| :--- | :--- |
| **Backend Framework** | Python 3.13, FastAPI, Pydantic, SQLAlchemy (Async), Uvicorn |
| **ML Runtime** | ONNX Runtime, Scikit-Learn, MLflow Registry |
| **Infrastructure** | Apache Kafka (aiokafka), Redis, PostgreSQL, MinIO/S3 |
| **Observability** | Prometheus, Grafana |
| **Frontend** | React 18, TypeScript, Tailwind CSS, TanStack Query, Vitest |
| **Tooling** | `uv` (Package Management), Ruff (Linting), Mypy (Types), Pytest (Testing), Docker |

---

## Deployment and Getting Started

### Production-ready Deployment (Docker)
Deploy the entire stack—including the API, Dashboard, Kafka, and Monitoring—with a single command:

```bash
docker compose up -d --build
```

-   **API Gateway**: `http://localhost:8000`
-   **Control Plane Dashboard**: `http://localhost:5173`
-   **Grafana (Observability)**: `http://localhost:3000` (Credentials: Admin/admin)
-   **Prometheus (Metrics)**: `http://localhost:9090`

### Local Development Environment

1.  **Environment Setup**: We use `uv` for ultra-fast dependency management:
    ```bash
    uv sync
    ```

2.  **Generate Test Artifacts & DB Seed**:
    ```bash
    uv run scripts/train_model.py
    uv run scripts/seed_features.py
    ```

3.  **Start Inference Service**:
    ```bash
    uv run python -m src.infrastructure.http.fastapi_server
    ```

---

## Quality Assurance

We maintain 100% compliance with strict quality gates enforced via CI/CD. Run these locally before pushing:

```bash
# Backend Quality Gate
uv run ruff check .
uv run mypy . --explicit-package-bases
uv run pytest tests/

# Frontend Quality Gate
cd frontend
npm ci
npm run dev:test     # Run isolated Vitest suites
```

---

## Benchmarking

Performance tests evaluate latency bottlenecks across the inference pipeline:
```bash
uv run python benchmarks/latency_benchmark.py
uv run python benchmarks/throughput_benchmark.py
```

---

## Author
**Võ Thành Nguyễn**
-   Email: nguyenvothanh04@gmail.com
-   GitHub: @vtnguyen04
-   Repository: phoenix_ML
