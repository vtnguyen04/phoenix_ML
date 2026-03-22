# Troubleshooting Guide

## Lỗi thường gặp

### 1. `mlService.getModels is not a function`

**Nguyên nhân**: Frontend import sai hoặc MLService chưa được instantiate.

**Fix**:
```typescript
// ❌ Wrong
import mlService from './api/mlService';
mlService.getModels();

// ✅ Correct
import { MLService } from './api/mlService';
const service = new MLService();
service.getModels();
```

### 2. `ModelNotFoundError: Model 'credit-risk' not found`

**Nguyên nhân**: Model chưa được train hoặc chưa register.

**Fix**:
```bash
# Generate datasets and train models
uv run python scripts/generate_datasets.py
uv run dvc repro

# Hoặc train specific model
uv run python examples/credit_risk/train.py
```

### 3. `ConnectionRefusedError: Kafka not available`

**Nguyên nhân**: Kafka container chưa start hoặc port sai.

**Fix**:
```bash
# Check Kafka status
docker compose ps kafka

# Restart Kafka
docker compose restart kafka

# Verify
docker compose logs kafka | tail -20
```

**Note**: Platform works **without Kafka** — producer/consumer fallback to no-op mode.

### 4. `ConnectionRefusedError: Redis` hoặc `PostgreSQL`

**Fix**:
```bash
# Check all services
docker compose ps

# Restart specific service
docker compose restart redis
docker compose restart postgres

# Check logs
docker compose logs redis
docker compose logs postgres
```

### 5. `onnxruntime.InferenceSession` fails

**Nguyên nhân**: model.onnx file corrupt hoặc chưa tồn tại.

**Fix**:
```bash
# Check model exists
ls -la models/credit_risk/v1/model.onnx

# Regenerate
uv run python examples/credit_risk/train.py

# Hoặc generate mock model (CI/testing)
uv run python -c "from src.shared.utils.model_generator import generate_simple_onnx; generate_simple_onnx('models/credit_risk/v1/model.onnx', 7)"
```

### 6. `alembic` migration errors

```bash
# Check current state
uv run alembic current

# Reset (drop all + recreate)
uv run alembic downgrade base
uv run alembic upgrade head
```

### 7. Frontend build errors

```bash
# Clear cache
cd frontend
rm -rf node_modules dist
npm install
npm run dev
```

### 8. `ruff check` hoặc `mypy` errors

```bash
# Auto-fix ruff
uv run ruff check . --fix
uv run ruff format .

# Check specific file
uv run mypy src/infrastructure/bootstrap/container.py
```

### 9. Docker "port already in use"

```bash
# Find process on port
lsof -i :8001  # or whatever port

# Kill process
kill -9 <PID>

# Or change port in .env
API_PORT=8002
```

### 10. gRPC connection refused

```bash
# Verify gRPC server started
docker compose logs phoenix-api | grep gRPC

# Test with grpcurl
grpcurl -plaintext localhost:50051 list
```

## Port Mapping Reference

| Port | Service | Internal | Protocol |
|------|---------|----------|----------|
| **5174** | Frontend | 5173 | HTTP |
| **8001** | API | 8000 | HTTP |
| **50051** | gRPC | 50051 | gRPC |
| **5433** | PostgreSQL | 5432 | TCP |
| **6380** | Redis | 6379 | TCP |
| **9094** | Kafka | 9092 | TCP |
| **8082** | Kafka UI | 8080 | HTTP |
| **5001** | MLflow | 5000 | HTTP |
| **9091** | Prometheus | 9090 | HTTP |
| **3001** | Grafana | 3000 | HTTP |
| **16686** | Jaeger | 16686 | HTTP |
| **9000** | MinIO API | 9000 | HTTP |
| **9001** | MinIO Console | 9001 | HTTP |
| **8080** | Airflow | 8080 | HTTP |

## Debug Commands

```bash
# Check all services
docker compose ps

# View API logs (live)
docker compose logs -f phoenix-api

# Enter API container shell
docker compose exec phoenix-api bash

# Run pytest inside container
docker compose exec phoenix-api uv run pytest tests/ -v

# Check database
docker compose exec postgres psql -U phoenix -d phoenix -c "SELECT * FROM models;"

# Check Redis
docker compose exec redis redis-cli keys "features:*"

# Check Kafka topics
docker compose exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Check endpoint health
curl -s http://localhost:8001/health | python -m json.tool

# Test prediction
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk", "features": [0.5, 1.2, 0.8, 3.4, 0.1, 2.5, 1.0]}' \
  | python -m json.tool
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://phoenix:phoenix@postgres:5432/phoenix` | PostgreSQL connection |
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `KAFKA_URL` | `kafka:9092` | Kafka bootstrap servers |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `DEFAULT_MODEL_ID` | `credit-risk` | Default model to load |
| `DEFAULT_MODEL_VERSION` | `v1` | Default model version |
| `MODEL_CONFIG_DIR` | `model_configs` | YAML config directory |
| `INFERENCE_ENGINE` | `onnx` | Engine: onnx, tensorrt, triton |
| `MONITORING_INTERVAL_SECONDS` | `30` | Drift check interval |
| `DRIFT_THRESHOLD` | `0.3` | Default drift threshold |
| `ARTIFACT_STORAGE_DIR` | `./artifacts` | Local artifact storage path |
| `USE_REDIS` | `true` | Use Redis for feature store |

## Known Limitations

| Limitation | Workaround |
|-----------|-----------|
| gRPC health check may timeout on cold start | Wait 5s after API start |
| Kafka consumer rebalance on restart | Consumer group auto-rebalances |
| MLflow concurrent writes | Use separate experiment names |
| Large batch size may OOM | Set `BATCH_MAX_SIZE` appropriately |
| ONNX Runtime GPU requires CUDA | Falls back to CPU automatically |

---
*Document Status: v4.0 — Updated March 2026*
