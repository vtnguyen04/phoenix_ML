# Deployment Guide: Docker Compose Stack

## Service Architecture

The platform consists of **9 interconnected services**:

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

## Quick Start

```bash
# Start all services
docker compose up -d

# Train champion model
uv run python scripts/train_model.py

# Seed feature store
uv run python scripts/seed_features.py

# Verify
curl http://localhost:8001/api/health
open http://localhost:5174      # Dashboard
open http://localhost:3001      # Grafana
open http://localhost:9001      # MinIO console
```

## DVC Pipeline

```bash
# Run full ML pipeline (train + seed)
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
| `./grafana/provisioning` | `/etc/grafana/provisioning` | Grafana datasources + dashboards |
| `minio_data` | `/data` | MinIO persistent storage |

## Production Scaling

```bash
# Scale API horizontally
docker compose up -d --scale api=3

# GPU acceleration: update Dockerfile with onnxruntime-gpu
# Kubernetes: Helm charts available in deploy/helm/phoenix-ml/
```

---
*Updated March 2026*