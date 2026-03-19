# Deployment Guide: Docker Compose Stack

## Service Architecture

The platform consists of **14+ interconnected services** across two compose files:

### Core Services (`compose.yaml`)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **api** | Custom (FastAPI) | 8001→8000 | ML inference, monitoring, model management |
| **frontend** | Custom (React/Vite) | 5174→5173 | Dashboard with embedded Grafana |
| **redis** | redis:alpine | 6380→6379 | Online feature store |
| **db** | postgres:15-alpine | 5433→5432 | Model metadata, prediction logs, drift reports |
| **kafka** | apache/kafka:latest | 9094→9092 | Async event streaming (KRaft mode) |
| **prometheus** | prom/prometheus | 9091→9090 | Metrics scraping |
| **grafana** | grafana/grafana | 3001→3000 | Dashboards (embedding enabled) |
| **minio** | minio/minio | 9000, 9001 | S3-compatible storage (DVC remote) |
| **createbuckets** | minio/mc | — | Auto-creates `dvc` bucket on startup |
| **mlflow** | Custom | 5000→5000 | Experiment tracking |
| **jaeger** | jaegertracing/all-in-one | 16686 | Distributed tracing (OpenTelemetry) |

### Airflow Services (`docker-compose.airflow.yaml`)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **airflow-webserver** | Custom | 8080→8080 | DAG UI (admin/admin) |
| **airflow-scheduler** | Custom | — | Task scheduling |
| **airflow-init** | Custom | — | DB migration + admin user |

## Quick Start

```bash
# Start all core services
docker compose up -d --build

# Start Airflow
docker compose -f docker-compose.airflow.yaml up -d

# Train models (run any/all)
uv run python examples/credit_risk/train.py
uv run python examples/house_price/train.py
uv run python examples/fraud_detection/train.py
uv run python examples/image_classification/train.py

# Seed feature store
uv run python scripts/seed_features.py

# Verify
curl http://localhost:8001/health
open http://localhost:5174      # Dashboard
open http://localhost:3001      # Grafana
open http://localhost:9001      # MinIO console
open http://localhost:8080      # Airflow (admin/admin)
open http://localhost:5000      # MLflow
```

## DVC Pipeline

```bash
# Run full ML pipeline (train all models + seed)
uv run dvc repro

# Push artifacts to MinIO
uv run dvc push
```

## Environment Variables

### API Service
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `KAFKA_URL` | `kafka:9092` | Kafka broker |
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@db:5432/phoenix` | Postgres connection |
| `USE_REDIS` | `true` | Enable Redis feature store |
| `DEFAULT_MODEL_ID` | `credit-risk` | Default model for scripts |
| `API_URL` | `http://localhost:8000` | API base URL for scripts |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `JAEGER_ENDPOINT` | `http://jaeger:4317` | OpenTelemetry OTLP endpoint |

### Grafana
| Variable | Value | Description |
|----------|-------|-------------|
| `GF_SECURITY_ALLOW_EMBEDDING` | `true` | Allow iframe embedding in frontend |
| `GF_AUTH_ANONYMOUS_ENABLED` | `true` | No login required for embed |
| `GF_AUTH_ANONYMOUS_ORG_ROLE` | `Viewer` | Anonymous access level |

## Volumes

| Volume/Mount | Container Path | Purpose |
|-------------|---------------|---------|
| `./models` | `/app/models` | ONNX model artifacts |
| `./src` | `/app/src` | Hot-reload source code |
| `./data` | `/app/data` | Reference data for drift detection |
| `./examples` | `/app/examples` | Training scripts (Airflow access) |
| `./model_configs` | `/app/model_configs` | Model configurations (Airflow access) |
| `./grafana/provisioning` | `/etc/grafana/provisioning` | Grafana datasources + dashboards |
| `minio_data` | `/data` | MinIO persistent storage |

## Production Scaling

```bash
# Scale API horizontally
docker compose up -d --scale api=3

# GPU acceleration: update Dockerfile with onnxruntime-gpu
```

---
*Updated March 2026*