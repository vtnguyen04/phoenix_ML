# Deployment Guide: Docker Compose Stack

## Overview

Phoenix ML deploys via **Docker Compose** with 14+ containers, or **Kubernetes** via Helm chart.

## Quick Start

```bash
# 1. Clone & config
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML
cp .env.example .env

# 2. Start infrastructure services
docker compose up -d

# 3. Start Airflow (optional)
docker compose -f docker-compose.airflow.yaml up -d

# 4. Generate data & train models
uv run python scripts/generate_datasets.py
uv run python examples/credit_risk/train.py

# 5. Verify
curl http://localhost:8001/health
# → {"status": "healthy", "version": "1.0.0"}
```

## Service Architecture

### docker-compose.yaml — Core Services

```mermaid
graph LR
    subgraph "Frontend"
        FE["phoenix-frontend<br/>:5174"]
    end
    
    subgraph "API"
        API["phoenix-api<br/>:8001 + :50051 gRPC"]
    end
    
    subgraph "Data"
        PG["postgres<br/>:5433"]
        RD["redis<br/>:6380"]
    end
    
    subgraph "Messaging"
        KF["kafka<br/>:9094"]
        KUI["kafka-ui<br/>:8082"]
    end
    
    subgraph "ML"
        MLF["mlflow<br/>:5001"]
        MINIO["minio<br/>:9000"]
    end
    
    subgraph "Monitoring"
        PROM["prometheus<br/>:9091"]
        GRAF["grafana<br/>:3001"]
        JAEG["jaeger<br/>:16686"]
    end
    
    FE --> API
    API --> PG
    API --> RD
    API --> KF
    API --> MLF
    API --> PROM
    API --> JAEG
    KUI --> KF
    PROM --> GRAF
    MLF --> MINIO
```

### Service Details

| Service | Image | Internal Port | External Port | Volume Mounts | Health Check |
|---------|-------|--------------|---------------|---------------|-------------|
| `phoenix-api` | Build from `Dockerfile` | 8000 | 8001 | `./src`, `./models`, `./data`, `./model_configs` | `/health` |
| `phoenix-frontend` | Build from `Dockerfile.frontend` | 5173 | 5174 | `./frontend/src` | — |
| `postgres` | `postgres:15-alpine` | 5432 | 5433 | `pgdata` volume | `pg_isready` |
| `redis` | `redis:7-alpine` | 6379 | 6380 | `redis_data` volume | `redis-cli ping` |
| `kafka` | `apache/kafka:latest` | 9092 | 9094 | `kafka_data` volume | — |
| `kafka-ui` | `provectuslabs/kafka-ui` | 8080 | 8082 | — | — |
| `mlflow` | `ghcr.io/mlflow/mlflow` | 5000 | 5001 | `mlflow_data` volume | — |
| `prometheus` | `prom/prometheus` | 9090 | 9091 | `./prometheus.yml` | — |
| `grafana` | `grafana/grafana` | 3000 | 3001 | `./grafana/` | — |
| `jaeger` | `jaegertracing/all-in-one` | 16686 | 16686 | — | — |
| `minio` | `minio/minio` | 9000/9001 | 9000/9001 | `minio_data` volume | — |

### docker-compose.airflow.yaml — Airflow Stack

| Service | Image | Port | Purpose |
|---------|-------|------|----------|
| `airflow-webserver` | Build from `Dockerfile.airflow` | 8080 | Airflow Web UI |
| `airflow-scheduler` | Build from `Dockerfile.airflow` | — | DAG scheduling |
| `airflow-init` | Build from `Dockerfile.airflow` | — | Init DB + admin user (admin/admin) |
| `postgres-airflow` | `postgres:15-alpine` | 5432 | Airflow metadata DB |

## Environment Variables (.env)

```bash
# ── Database ──
DATABASE_URL=postgresql+asyncpg://phoenix:phoenix@postgres:5432/phoenix
TEST_DATABASE_URL=sqlite+aiosqlite:///./phoenix_test.db

# ── Redis ──
REDIS_URL=redis://redis:6379

# ── Kafka ──
KAFKA_URL=kafka:9092

# ── MLflow ──
MLFLOW_TRACKING_URI=http://mlflow:5000

# ── Model Config ──
DEFAULT_MODEL_ID=credit-risk
DEFAULT_MODEL_VERSION=v1
MODEL_CONFIG_DIR=model_configs
INFERENCE_ENGINE=onnx

# ── Monitoring ──
MONITORING_INTERVAL_SECONDS=30
DRIFT_THRESHOLD=0.3

# ── Storage ──
ARTIFACT_STORAGE_DIR=./artifacts
CACHE_DIR=./cache

# ── Ports ──
API_PORT=8001
FRONTEND_PORT=5174
```

## Makefile Commands

```bash
make up          # docker compose up -d
make down        # docker compose down
make build       # docker compose build
make restart     # down + up
make logs        # docker compose logs -f
make ps          # docker compose ps
make clean       # down -v (remove volumes)
make test        # uv run pytest
make api-logs    # docker compose logs -f phoenix-api
make frontend-logs  # docker compose logs -f phoenix-frontend
```

## Training Pipeline

Training is managed via **Airflow DAG** (`dags/retrain_pipeline.py`) or direct scripts:

```bash
# Generate data + train all models
uv run python scripts/generate_datasets.py
uv run python examples/credit_risk/train.py
uv run python examples/house_price/train.py
uv run python examples/fraud_detection/train.py
uv run python examples/image_classification/train.py
uv run python examples/sentiment/train.py

# Run end-to-end simulation
uv run python scripts/simulate_pipeline.py --fast

# Or trigger via Airflow UI (http://localhost:8080)
```

## Kubernetes Deployment (Helm)

```bash
# Install via Helm
helm install phoenix-ml deploy/helm/phoenix-ml/

# Custom values
helm install phoenix-ml deploy/helm/phoenix-ml/ \
  --set replicaCount=3 \
  --set image.tag=v1.0.0 \
  --set resources.limits.memory=2Gi

# Upgrade
helm upgrade phoenix-ml deploy/helm/phoenix-ml/

# Uninstall
helm uninstall phoenix-ml
```

### Helm Chart Structure

```
deploy/helm/phoenix-ml/
├── Chart.yaml              # name: phoenix-ml, version: 0.1.0
├── values.yaml             # Default: 1 replica, 512Mi memory
└── templates/
    ├── deployment.yaml     # Pod spec, health probes, env vars
    ├── service.yaml        # ClusterIP on port 8000
    ├── ingress.yaml        # External access, TLS
    └── hpa.yaml            # Auto-scale: 1-10 replicas, 80% CPU target
```

## Database Migrations

```bash
# Apply all migrations
uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "add new table"

# Rollback last migration
uv run alembic downgrade -1

# View current version
uv run alembic current
```

### Schema

```
models
├── id (VARCHAR, PK)
├── version (VARCHAR, PK)
├── uri (VARCHAR)
├── framework (VARCHAR)
├── stage (VARCHAR: champion/challenger/retired/archived)
├── metadata_json (JSONB)
├── metrics_json (JSONB)
├── is_active (BOOLEAN)
└── created_at (TIMESTAMP)

prediction_logs
├── id (UUID, PK)
├── model_id (VARCHAR)
├── model_version (VARCHAR)
├── features_json (JSONB)
├── result (FLOAT)
├── confidence (FLOAT)
├── latency_ms (FLOAT)
├── ground_truth (FLOAT, nullable)
└── timestamp (TIMESTAMP)

drift_reports
├── id (UUID, PK)
├── model_id (VARCHAR)
├── feature_name (VARCHAR)
├── method (VARCHAR: ks/psi/chi2)
├── score (FLOAT)
├── is_drifted (BOOLEAN)
├── threshold (FLOAT)
└── timestamp (TIMESTAMP)
```

## Grafana (Auto-provisioned)

Dashboards and datasources auto-provisioned on startup:

- **Datasource**: Prometheus (`http://prometheus:9090`)
- **Dashboard**: `phoenix-ml.json` — panels: prediction count, latency histogram, drift score, model accuracy

Access: `http://localhost:3001` (admin/admin)

## Port Summary

| Port | Service | Protocol |
|------|---------|----------|
| 5174 | Frontend (React) | HTTP |
| 8001 | API (FastAPI) | HTTP |
| 50051 | gRPC Server | gRPC |
| 5433 | PostgreSQL | TCP |
| 6380 | Redis | TCP |
| 9094 | Kafka | TCP |
| 8082 | Kafka UI | HTTP |
| 5001 | MLflow | HTTP |
| 9091 | Prometheus | HTTP |
| 3001 | Grafana | HTTP |
| 16686 | Jaeger | HTTP |
| 9000 | MinIO (API) | HTTP |
| 9001 | MinIO (Console) | HTTP |
| 8080 | Airflow | HTTP |

---
*Document Status: v4.0 — Updated March 2026*