# 🚀 Phoenix ML — Quick Start Guide

## Prerequisites

- Docker Desktop installed and running
- Python 3.11+ with `uv` package manager
- Git
- ~4 GB free RAM (for all services)

## Step 1: Start the Stack

```bash
# Clone and enter the project
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML

# Start all services (first time takes ~5 min to pull images)
docker compose up -d --build

# (Optional) Start Airflow for self-healing pipeline
docker compose -f docker-compose.airflow.yaml up -d

# Check all services are healthy
docker compose ps
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
| **Airflow** | 8080 | http://localhost:8080 (admin/admin) |

## Step 2: Train Models

```bash
# Install Python deps
uv sync

# Train any / all models:
uv run python examples/credit_risk/train.py          # Classification (30 features)
uv run python examples/house_price/train.py           # Regression (8 features)
uv run python examples/fraud_detection/train.py       # XGBoost (12 features)
uv run python examples/image_classification/train.py  # MLP (784 features)

# Or: train all via DVC
uv run dvc repro

# Seed feature store (for entity_id-based predictions)
uv run python scripts/seed_features.py
```

## Step 3: Test Predictions

```bash
# Restart API to pick up trained models
docker compose restart api

# Test a prediction (credit risk, 30 features)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "model_version": "v1", "features": [0.5, 0.3, 0.8, 1.2, -0.4, 0.7, -1.1, 0.3, 0.9, -0.2, 0.6, -0.8, 1.4, 0.1, -0.5, 0.4, 0.3, -0.1, 0.8, -0.6, 0.2, 1.0, -0.3, 0.5, 0.7, -0.4, 0.9, 0.1, -0.7, 0.3]}'

# Test via feature store
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "entity_id": "customer-0001"}'

# Batch prediction
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "house-price", "model_version": "v1", "batch": [[3.5, 25, 5.0, 1.0, 1500, 3.0, 37.5, -122.0], [8.0, 10, 6.5, 1.2, 800, 2.5, 34.0, -118.5]]}'
```

## Step 4: Simulation Scripts

```bash
# Simulate production traffic
DEFAULT_MODEL_ID=credit-risk uv run python scripts/simulate_traffic.py

# Simulate drifted data (triggers drift detection)
DEFAULT_MODEL_ID=credit-risk uv run python scripts/simulate_drift.py

# Full production simulation
uv run python scripts/run_production_simulation.py
```

## Step 5: Open Dashboard

Visit **http://localhost:5174** to see:
- 📊 Model metrics (accuracy, F1, precision, recall)
- 🔮 Live prediction panel
- 📈 Drift detection reports
- 🏥 Service health status

## Local Development (without Docker)

```bash
# Install deps
uv sync

# Run API on port 8000
uv run uvicorn src.infrastructure.http.fastapi_server:app --reload

# Or via CLI entry point (after pip install -e .)
phoenix-serve

# Run frontend
cd frontend && npm install && npm run dev
```

## Useful Commands

```bash
docker compose ps             # Check service status
docker compose logs -f api    # Follow API logs
docker compose restart api    # Restart API
docker compose down           # Stop all services
docker compose down -v        # Stop + remove volumes
uv run ruff check .           # Lint check
uv run pytest tests/unit/     # Run tests
```

## Troubleshooting

**API unhealthy?** Check logs: `docker compose logs api`
**Model not loading?** Ensure `models/credit_risk/v1/model.onnx` exists (run Step 2)
**Frontend blank?** API must be healthy first. Check `curl http://localhost:8001/health`
