# 🚀 Phoenix ML — Quick Start Guide

## Prerequisites

- Docker Desktop installed and running
- Git
- ~4 GB free RAM (for all services)

## Step 1: Start the Stack

```bash
# Clone and enter the project
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML

# Start all services (first time takes ~5 min to pull images)
make up-build
# or: docker compose up -d --build

# Check all services are healthy
make ps
# or: docker compose ps
```

Expected services:
| Service | Port | URL |
|---------|------|-----|
| **Frontend Dashboard** | 5174 | http://localhost:5174 |
| **API (FastAPI)** | 8001 | http://localhost:8001/health |
| **gRPC** | 50051 | — |
| **PostgreSQL** | 5433 | — |
| **Redis** | 6380 | — |
| **Kafka** | 9094 | — |
| **MLflow** | 5000 | http://localhost:5000 |
| **Prometheus** | 9091 | http://localhost:9091 |
| **Grafana** | 3001 | http://localhost:3001 |
| **Jaeger** | 16686 | http://localhost:16686 |
| **MinIO** | 9000/9001 | http://localhost:9001 |

## Step 2: Train the Model

```bash
# Train XGBoost on German Credit → export ONNX + metrics + reference data
python scripts/train_model.py

# Seed 100 real feature records into data/reference_features.json
python scripts/seed_features.py
```

## Step 3: Test Predictions

```bash
# The API auto-loads the trained model on startup.
# Restart API to pick up new model:
docker compose restart api

# Test a prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "entity_id": "customer-0001"}'
```

## Step 4: Full Data Pipeline (with Docker services)

```bash
# Ingest real data into Redis + Postgres (requires docker services running)
PYTHONPATH=. python scripts/ingest_real_data.py

# Simulate traffic for drift detection
PYTHONPATH=. python scripts/simulate_traffic.py
```

## Step 5: Open Dashboard

Visit **http://localhost:5174** to see:
- 📊 Model metrics (accuracy, F1, precision, recall)
- 🔮 Live prediction panel
- 📈 Drift detection reports
- 🏥 Service health status

## Useful Commands

```bash
make logs           # Follow all logs
make api-logs       # Follow API logs only
make db-shell       # Open PostgreSQL shell
make redis-cli      # Open Redis CLI
make down           # Stop all services
make down-clean     # Stop + remove all data volumes
make lint           # Run ruff + mypy
make test           # Run pytest
```

## Troubleshooting

**API unhealthy?** Check logs: `make api-logs`
**Model not loading?** Ensure `models/credit_risk/v1/model.onnx` exists (run Step 2)
**Frontend blank?** API must be healthy first. Check `curl http://localhost:8001/health`
