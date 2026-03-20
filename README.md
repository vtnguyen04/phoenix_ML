<div align="center">

<img src="docs/assets/logo.png" alt="Phoenix ML Logo" width="280"/>

# Phoenix ML Platform

### High-Throughput, Low-Latency Real-time ML Inference System with Autonomous Self-Healing

[![CI/CD](https://github.com/vtnguyen04/phoenix_ML/actions/workflows/ci.yaml/badge.svg)](https://github.com/vtnguyen04/phoenix_ML/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React_18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[📖 Documentation](https://vtnguyen04.github.io/phoenix_ML/)** · **[🚀 Quick Start](#-quick-start)** · **[📊 Architecture](#-system-architecture)** · **[🧪 Testing](#-quality-assurance)** · **[🛠️ Customization](docs/guides/customization.md)**

</div>

---

## 📌 What is Phoenix ML?

Phoenix ML is a **production-grade, model-agnostic ML inference platform** that goes beyond simple model serving. It combines real-time inference with autonomous monitoring, drift detection, and self-healing capabilities — all built with **Domain-Driven Design (DDD)** and **Clean Architecture** principles.

<div align="center">

![System Architecture](docs/assets/system-architecture.png)

</div>

### ✨ Key Capabilities

| Capability | Description |
|-----------|-------------|
| ⚡ **Real-time Inference** | Sub-50ms p99 latency with ONNX Runtime, TensorRT, and Triton support |
| 🔄 **Self-Healing** | Drift detection → Airflow pipeline: alert → rollback → retrain → log → deploy |
| 🎯 **A/B Testing** | Dynamic model routing with Champion/Challenger traffic splitting |
| 🛡️ **Circuit Breaker** | Fault tolerance with automatic failover and recovery |
| 📊 **Full Observability** | Prometheus metrics, Grafana dashboards, Jaeger distributed tracing |
| 🔬 **Anomaly Detection** | Real-time monitoring for prediction anomalies, latency spikes, error rates |
| 📦 **DVC Pipelines** | Reproducible model training with versioned data and artifacts |
| 🌀 **Airflow Orchestration** | 5-task self-healing DAG with max_active_runs=1 deduplication |
| 📦 **Batch Prediction** | `/predict/batch` endpoint with concurrent processing |
| 🧩 **Model-Agnostic** | Supports scikit-learn, XGBoost, MLP — any ONNX-exportable framework |

---

## 🧩 Model Examples

The platform ships with **4 production-ready examples** demonstrating model-agnostic capabilities:

| Example | ML Framework | Task | Features | Accuracy |
|---------|-------------|------|----------|----------|
| **Credit Risk** | scikit-learn (GBClassifier) | Binary Classification | 30 (tabular) | 78.5% |
| **House Price** | scikit-learn (Ridge) | Regression | 8 (tabular) | R² 0.61 |
| **Fraud Detection** | XGBoost | Imbalanced Classification | 12 (tabular) | 98.2% |
| **Image Classification** | sklearn MLP (256→128) | Multi-class (10 classes) | 784 (28×28 pixels) | 87.0% |

Each example lives in `examples/<problem>/train.py` with a corresponding `model_configs/<name>.yaml`.

Adding your own model:
```bash
# 1. Create training script
examples/my_problem/train.py     # implement train_and_export()

# 2. Create model config
model_configs/my-model.yaml      # model_id, paths, metadata

# 3. Train
uv run python examples/my_problem/train.py

# 4. Serve
uv run uvicorn src.infrastructure.http.fastapi_server:app --reload
```

---

## 🏗️ System Architecture

### Inference Pipeline

```mermaid
graph LR
    Client([🌐 Client]) -->|REST/gRPC| Gateway[⚡ API Gateway<br/>FastAPI]
    Gateway --> Pipeline[🔗 Request Pipeline<br/>Chain of Responsibility]
    Pipeline --> Router{🎯 Router<br/>A/B · Canary · Shadow}

    Router --> CB1[🛡️ Circuit<br/>Breaker]
    CB1 --> ONNX[ONNX Runtime]

    Router --> CB2[🛡️ Circuit<br/>Breaker]
    CB2 --> TRT[TensorRT]

    Router --> CB3[🛡️ Circuit<br/>Breaker]
    CB3 --> Triton[Triton Server]

    Gateway -.->|Async| Kafka[(📨 Kafka)]
    Gateway -->|MGET| Redis[(🔴 Redis<br/>Features)]
    Gateway -->|Query| PG[(🐘 PostgreSQL)]

    style Client fill:#667eea,stroke:#764ba2,color:#fff
    style Gateway fill:#f093fb,stroke:#f5576c,color:#fff
    style Router fill:#4facfe,stroke:#00f2fe,color:#fff
    style ONNX fill:#43e97b,stroke:#38f9d7,color:#000
    style TRT fill:#43e97b,stroke:#38f9d7,color:#000
    style Triton fill:#43e97b,stroke:#38f9d7,color:#000
```

### Self-Healing MLOps Loop

```mermaid
graph TD
    Train["🏋️ Model Training<br/>DVC Pipeline"] --> Deploy["📦 Deployment<br/>ONNX + Docker"]
    Deploy --> Serve["⚡ Real-time Inference<br/>FastAPI + Routing"]
    Serve --> Monitor["📊 Monitoring<br/>Every 30s"]
    Monitor --> Anomaly["🔍 Drift Detection<br/>KS · PSI · Chi² · Wasserstein"]
    Anomaly --> Decision{"Drift<br/>Detected?"}
    Decision -->|No| Serve
    Decision -->|"Yes (deduped)"| Airflow["🌀 Airflow self_healing_pipeline<br/>max_active_runs=1"]
    Airflow --> T1["🚨 1. Send Alert<br/>Webhook Notify"]
    T1 --> T2["⏪ 2. Rollback<br/>Archive Challengers"]
    T2 --> T3["🏋️ 3. Train Model<br/>ONNX Export"]
    T3 --> T4["📈 4. Log MLflow<br/>Metrics + Params"]
    T4 --> T5["🗄️ 5. Register<br/>Challenger in Postgres"]
    T5 --> Deploy
    T2 -.->|champion continues| Serve

    style Train fill:#667eea,stroke:#764ba2,color:#fff
    style Deploy fill:#764ba2,stroke:#667eea,color:#fff
    style Serve fill:#f093fb,stroke:#f5576c,color:#fff
    style Monitor fill:#4facfe,stroke:#00f2fe,color:#fff
    style Anomaly fill:#fa709a,stroke:#fee140,color:#fff
    style Decision fill:#ffecd2,stroke:#fcb69f,color:#000
    style Airflow fill:#017cee,stroke:#00c7b7,color:#fff
    style T1 fill:#f5576c,stroke:#ff6b6b,color:#fff
    style T2 fill:#ff9a9e,stroke:#fad0c4,color:#000
    style T3 fill:#43e97b,stroke:#38f9d7,color:#000
    style T4 fill:#43e97b,stroke:#38f9d7,color:#000
    style T5 fill:#fa709a,stroke:#fee140,color:#fff
```

---

## 📓 Interactive Notebooks

| Notebook | Description |
|----------|-------------|
| [01 — Inference Demo](notebooks/01_inference_demo.ipynb) | Real-time predictions, latency benchmarks (p50/p95/p99) |
| [02 — Drift Detection](notebooks/02_drift_detection_demo.ipynb) | KS test, PSI, Wasserstein distance with visualization |
| [03 — Self-Healing](notebooks/03_self_healing_demo.ipynb) | Trigger Airflow DAG, monitor tasks, verify new model |
| [04 — A/B Testing](notebooks/04_ab_testing_demo.ipynb) | Champion/Challenger lifecycle, rollback, traffic simulation |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Inference** | ONNX Runtime · TensorRT · Triton Inference Server |
| **Backend** | Python 3.11+ · FastAPI · gRPC · Pydantic v2 · SQLAlchemy 2.0 Async |
| **Data** | Redis (features) · PostgreSQL (metadata) · Apache Kafka (events) · MinIO/S3 (artifacts) |
| **ML Frameworks** | Scikit-Learn · XGBoost · ONNX · skl2onnx · onnxmltools |
| **MLOps** | DVC (data versioning) · MLflow (experiment tracking) · Apache Airflow (orchestration) |
| **Observability** | Prometheus · Grafana · Jaeger (OpenTelemetry) |
| **Frontend** | React 18 · TypeScript · Vite · Vanilla CSS · Vitest |
| **Infrastructure** | Docker Compose (14+ services) · GitHub Actions CI · `uv` package manager · Alembic migrations |
| **Testing** | Pytest (87% coverage) · Vitest (104 tests) · ESLint · Ruff · Mypy (142 source files, 0 errors) |

---

## 🏛️ Architecture & Design Patterns

### Clean Architecture (DDD)

```
src/
├── domain/                    # 🧠 Pure business logic (zero framework deps)
│   ├── inference/             #    Model, Prediction, InferenceEngine interface
│   ├── feature_store/         #    FeatureRegistry, FeatureStore interface
│   ├── model_registry/        #    ModelRepository, ArtifactStorage interface
│   ├── monitoring/            #    DriftReport, DriftCalculator, AnomalyDetector
│   └── training/              #    TrainingJob, TrainerPlugin, DataLoaderPlugin
│
├── application/               # 🎯 Use-case orchestration (CQRS)
│   ├── commands/              #    PredictCommand, BatchPredictCommand, LoadModelCommand
│   ├── handlers/              #    PredictHandler, BatchPredictHandler, QueryHandlers
│   └── services/              #    MonitoringService
│
├── infrastructure/            # 🔌 Framework adapters
│   ├── http/                  #    FastAPI, gRPC, Routes, DI Container
│   ├── ml_engines/            #    ONNX, TensorRT, Triton implementations
│   ├── feature_store/         #    Redis, InMemory
│   ├── persistence/           #    Postgres, SQLite, InMemory repos
│   ├── messaging/             #    Kafka Producer/Consumer
│   ├── monitoring/            #    Prometheus, Jaeger, Alert Notifier
│   └── artifact_storage/      #    S3 (MinIO), Local
│
└── shared/                    # 🔧 Cross-cutting utilities
    ├── exceptions/            #    PhoenixBaseError hierarchy
    ├── interfaces/            #    EventPublisher, CacheBackend
    ├── ingestion/             #    DataCollector, IngestionService
    └── utils/                 #    ModelGenerator, helpers
```

### Design Patterns Implemented

| Pattern | Implementation | Purpose |
|---------|---------------|---------|
| **Strategy** | `RoutingStrategy` (ABTesting, Canary, Shadow) | Dynamic model traffic routing |
| **Circuit Breaker** | `CircuitBreaker` (Closed → Open → Half-Open) | Fault tolerance & auto-recovery |
| **Chain of Responsibility** | `RequestPipeline` (Validate → Cache → Feature → Infer) | Composable request processing |
| **Command/CQRS** | `PredictCommand` → `PredictHandler` | Separate read/write concerns |
| **Repository** | `ModelRepository`, `FeatureStore` | Data access abstraction |
| **Observer** | Kafka event bus | Async event propagation |
| **Dependency Injection** | `Container` class | Framework decoupling |

---

## 🚀 Quick Start

### Docker Compose (Full Stack — 14+ Services)

```bash
# Clone and start everything
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML

# Start core services (API, DB, Redis, Kafka, MLflow, Grafana, etc.)
docker compose up -d --build

# Start Airflow orchestration services
docker compose -f docker-compose.airflow.yaml up -d

# Train all models
uv run python examples/credit_risk/train.py
uv run python examples/house_price/train.py
uv run python examples/fraud_detection/train.py
uv run python examples/image_classification/train.py

# Or: reproduce the full DVC pipeline
uv run dvc repro
```

| Service | URL | Purpose |
|---------|-----|---------|
| 🔗 **API** | http://localhost:8001 | ML inference + monitoring |
| 🖥️ **Dashboard** | http://localhost:5174 | React control plane |
| 📊 **Grafana** | http://localhost:3001 | Metrics visualization |
| 📈 **Prometheus** | http://localhost:9091 | Metrics collection |
| 💾 **MinIO** | http://localhost:9001 | Model artifact storage |
| 🔍 **Jaeger** | http://localhost:16686 | Distributed tracing |
| 🌀 **Airflow** | http://localhost:8080 | Pipeline orchestration (admin/admin) |
| 📈 **MLflow** | http://localhost:5000 | Experiment tracking |

### Make Predictions

```bash
# Single prediction (via raw features)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "model_version": "v1", "features": [0.5, 0.3, 0.8, 1.2, -0.4, 0.7, -1.1, 0.3, 0.9, -0.2, 0.6, -0.8, 1.4, 0.1, -0.5, 0.4, 0.3, -0.1, 0.8, -0.6, 0.2, 1.0, -0.3, 0.5, 0.7, -0.4, 0.9, 0.1, -0.7, 0.3]}'

# Single prediction (via feature store)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "entity_id": "customer-good"}'

# Batch prediction (multiple inputs)
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "model_version": "v1", "batch": [[0.5, 0.3, ...], [1.2, -0.4, ...]]}'
```

### Local Development

```bash
# Install dependencies
uv sync

# Run API server (port 8000)
uv run uvicorn src.infrastructure.http.fastapi_server:app --reload

# Or via CLI entry point (after pip install)
phoenix-serve

# Run frontend
cd frontend && npm install && npm run dev
```

---

## 🧪 Quality Assurance

### CI Pipeline

```bash
# All quality gates (last run: all pass ✅)
uv run ruff check src/ tests/ scripts/ dags/    # Linting — 0 errors
uv run mypy src/ --ignore-missing-imports        # Type checking — 0 errors (142 files)
uv run pytest tests/ --cov=src                   # Tests — 87% coverage

# Frontend tests
cd frontend && npx vitest run                    # 104 tests, 16 test files
npx eslint . --ext ts,tsx                        # ESLint — 0 warnings

# Load testing
uv run locust -f benchmarks/load_test.py --host http://localhost:8001
```

---

## 📊 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/feedback` | Submit ground truth |
| GET | `/models/{model_id}` | Get model info |
| POST | `/models/register` | Register new model version |
| POST | `/models/rollback` | Rollback challengers |
| GET | `/monitoring/drift/{model_id}` | Trigger drift check |
| GET | `/monitoring/reports/{model_id}` | Drift reports history |
| GET | `/monitoring/performance/{model_id}` | Performance metrics |
| GET | `/metrics` | Prometheus metrics |

---

## 📁 Repository Structure

```
phoenix-ml-platform/
├── src/                         # Backend (DDD layers, 131 files, 7k LOC)
│   ├── domain/                  # Business logic + interfaces
│   ├── application/             # Commands, handlers, DTOs
│   ├── infrastructure/          # FastAPI, ONNX, Postgres, Redis, Kafka
│   └── shared/                  # Exceptions, interfaces, utils
├── examples/                    # ML training examples
│   ├── credit_risk/             # GBClassifier (30 features)
│   ├── house_price/             # Ridge regression (8 features)
│   ├── fraud_detection/         # XGBoost (12 features)
│   └── image_classification/    # MLP 256→128 (784 features)
├── model_configs/               # YAML configs per model
├── frontend/                    # React + TypeScript dashboard
├── dags/                        # Airflow DAGs (self_healing_pipeline)
├── tests/                       # Unit / Integration / E2E
├── scripts/                     # Simulation, seeding, E2E test scripts
├── benchmarks/                  # Locust load tests
├── docs/                        # Architecture, ADRs, API ref, customization, monitoring, troubleshooting
├── grafana/                     # Provisioned dashboards
├── .github/workflows/           # CI/CD pipelines
├── dvc.yaml                     # ML pipeline stages
├── compose.yaml                 # Core services (14+ containers)
├── docker-compose.airflow.yaml  # Airflow orchestration stack
└── pyproject.toml               # Python dependencies + packaging
```

---

## 📜 Architecture Decision Records

| ADR | Decision | Rationale |
|-----|----------|-----------|
| [001](docs/adr/001-use-ddd-architecture.md) | DDD + Clean Architecture | Testability, flexibility, maintainability |
| [002](docs/adr/002-use-onnx-runtime.md) | ONNX Runtime standardization | Framework-agnostic, 2-10x inference speedup |
| [003](docs/adr/003-use-kafka-for-event-streaming.md) | Kafka for event streaming | Decouple inference from logging, durability |
| [004](docs/adr/004-observability-with-prometheus-grafana.md) | Prometheus + Grafana | Real-time metrics, dashboard-as-code |
| [005](docs/adr/005-dvc-data-versioning.md) | DVC + MinIO | Reproducible pipelines, S3-compatible storage |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup, code standards, and PR checklist.

```bash
git checkout -b feature/your-feature
git commit -m "feat(scope): description"
git push origin feature/your-feature
```

**Branch naming**: `feature/`, `fix/`, `docs/`, `refactor/`
**Commit style**: [Conventional Commits](https://www.conventionalcommits.org/)

---

<div align="center">

## 👤 Author

**Võ Thành Nguyễn**

[![GitHub](https://img.shields.io/badge/GitHub-@vtnguyen04-181717?logo=github)](https://github.com/vtnguyen04)
[![Email](https://img.shields.io/badge/Email-nguyenvothanh04@gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:nguyenvothanh04@gmail.com)

---

*Built with ❤️ using DDD, SOLID, and a passion for production-grade ML systems*

</div>
