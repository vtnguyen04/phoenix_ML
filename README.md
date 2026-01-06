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

## Project Structure

The project follows a modular structure reflecting the Inversion of Control principle:

```text
.
├── src/
│   ├── domain/              # Pure Logic Layer (Zero dependencies)
│   │   ├── inference/       # Core ML logic (Entities, Value Objects, Routing)
│   │   ├── feature_store/   # Feature retrieval interfaces
│   │   ├── model_registry/  # Artifact storage interfaces
│   │   └── monitoring/      # Statistical drift algorithms
│   ├── application/         # Orchestration Layer (Commands & Handlers)
│   │   ├── commands/        # Command objects (Predict, LoadModel)
│   │   ├── handlers/        # Logic orchestrators
│   │   └── services/        # Cross-context application services
│   ├── infrastructure/      # Implementation Layer (Adapters)
│   │   ├── http/            # FastAPI configuration and routes
│   │   ├── ml_engines/      # ONNX and TensorRT implementations
│   │   ├── messaging/       # Kafka producers and consumers
│   │   ├── persistence/     # PostgreSQL and Redis repositories
│   │   └── artifact_storage/# Local and S3 model loading
│   └── shared/              # Utilities and Base interfaces
├── frontend/                # React + TypeScript Control Plane
│   ├── src/api/             # Standardized API client
│   ├── src/components/      # Reusable UI components
│   └── src/types/           # Shared TypeScript interfaces
├── tests/                   # Comprehensive Test Suite
│   ├── unit/                # Per-layer isolation tests
│   └── integration/         # End-to-end flow verification
├── scripts/                 # MLOps Pipelines (Training, Simulation)
├── grafana/                 # Monitoring Dashboards (Provisioning)
├── .github/workflows/       # Full-stack CI/CD (Ruff, Mypy, Vitest, Docker)
├── Dockerfile               # Backend production build
├── Dockerfile.frontend      # Frontend production build
└── compose.yaml             # Multi-container stack orchestration
```

---

## Technology Stack

| Category | Component |
| :--- | :--- |
| **Backend Framework** | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy (Async) |
| **ML Runtime** | ONNX Runtime, Scikit-Learn |
| **Infrastructure** | Apache Kafka (aiokafka), Redis 7, PostgreSQL 15 |
| **Observability** | Prometheus, Grafana |
| **Frontend** | React 18, TypeScript, Tailwind CSS, TanStack Query |
| **Tooling** | Ruff (Linting), Mypy (Type Checking), Pytest (Testing), Docker |

---

## Deployment and Getting Started

### Production-ready Deployment (Docker)
Deploy the entire stack—including the API, Dashboard, Kafka, and Monitoring—with a single command:

```bash
docker compose up -d --build
```

-   **API Gateway**: http://localhost:8000
-   **Control Plane Dashboard**: http://localhost:5173
-   **Grafana (Observability)**: http://localhost:3000 (Credentials: Admin/admin)
-   **Prometheus (Metrics)**: http://localhost:9090

### Local Development Environment

1.  **Environment Setup**:
    ```bash
    uv sync
    ```

2.  **Generate Model Artifacts**:
    ```bash
    uv run scripts/train_model.py
    uv run scripts/train_challenger.py
    ```

3.  **Start Inference Service**:
    ```bash
    uv run python -m src.infrastructure.http.fastapi_server
    ```

---

## Quality Assurance

We maintain 100% compliance with strict quality gates enforced via CI/CD:

```bash
# Python Quality Gate
uv run ruff check .
uv run mypy . --explicit-package-bases
uv run pytest

# Frontend Quality Gate
cd frontend
npm ci
npx tsc --noEmit
npx eslint . --ext ts,tsx
```

---

## Author
**Võ Thành Nguyễn**
-   Email: nguyenvothanh04@gmail.com
-   GitHub: @vtnguyen04
-   Repository: phoenix_ML
