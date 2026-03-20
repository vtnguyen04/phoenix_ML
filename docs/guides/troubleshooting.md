# Troubleshooting Guide

Common issues and solutions for the Phoenix ML Platform.

---

## Startup Issues

### API container won't start

```bash
docker compose logs api
```

**Common causes**:

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Rebuild: `docker compose build api` |
| `Connection refused` to DB | Wait for Postgres: `docker compose up -d db && sleep 5` |
| `ONNX model not found` | Train models first: `uv run python examples/credit_risk/train.py` |
| Port 8001 in use | `lsof -i :8001` and kill the process |

### Airflow webserver not accessible

```bash
docker compose -f docker-compose.airflow.yaml logs airflow-webserver
```

- **DB not initialized**: Run `docker compose -f docker-compose.airflow.yaml up -d airflow-init` first
- **Default credentials**: admin/admin

### Grafana shows "No data"

1. Check Prometheus is running: `curl http://localhost:9091/-/healthy`
2. Check Prometheus targets: http://localhost:9091/targets
3. Verify API is being scraped: look for `api:8000` target
4. Ensure API has handled some predictions first

---

## Prediction Errors

### `"Model not found"`

```bash
# Check registered models
curl http://localhost:8001/models

# If empty, train a model:
uv run python examples/credit_risk/train.py
docker compose restart api
```

### `"INVALID_ARGUMENT: Got invalid dimensions for input"`

Your feature vector length doesn't match the model:

| Model | Expected Features |
|-------|------------------|
| credit-risk | 30 |
| fraud-detection | 12 |
| house-price | 8 |
| image-class | 784 |

### `"Circuit breaker is open"`

The model engine hit too many errors and tripped the circuit breaker.

```bash
# Wait 30 seconds for half-open recovery, or restart API
docker compose restart api
```

### Prediction returns `NaN` or unexpected values

- Input features may contain `NaN`, `Inf`, or out-of-range values
- ONNX models trained on positive features may fail with negative inputs
- Validate input data before sending

---

## Monitoring Issues

### Drift detection returns `400`

```json
{"detail": "Not enough prediction data for drift detection"}
```

**Solution**: Send more predictions first. Drift detection requires at least ~30 data points.

```bash
uv run python scripts/simulate_traffic.py
```

### Self-healing pipeline not triggering

1. Check Airflow DAG is unpaused:
   ```bash
   curl http://localhost:8080/api/v1/dags/self_healing_pipeline \
     -u admin:admin | python3 -c "import json,sys; print(json.load(sys.stdin)['is_paused'])"
   ```
   If `true`, unpause in Airflow UI.

2. Check `max_active_runs=1` — if a run is already in progress, new triggers are deduped.

3. Check monitoring service is running: look for `Drift detected` in API logs:
   ```bash
   docker compose logs api | grep "Drift detected"
   ```

### Alerts not firing

- Check cooldown period — alerts won't re-fire within `ALERT_COOLDOWN_SECONDS` (default: 300s)
- Check alert rules are registered (they're set up in the monitoring service initialization)
- Check `ALERT_WEBHOOK_URL` is set if expecting Slack notifications

---

## Infrastructure Issues

### PostgreSQL connection errors

```bash
# Check if Postgres is running
docker exec phoenix-postgres pg_isready

# Check connection
docker exec phoenix-postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1"

# Reset database (⚠️ destroys data)
docker compose down -v
docker compose up -d
```

### Redis connection errors

```bash
# Check if Redis is running
docker exec phoenix-redis redis-cli PING   # Should return PONG

# Check key count
docker exec phoenix-redis redis-cli DBSIZE

# Feature store empty?
uv run python scripts/seed_features.py
```

### Kafka consumer lag

```bash
# Check Kafka broker status
docker exec phoenix-kafka kafka-broker-api-versions.sh --bootstrap-server localhost:9092
```

### MinIO / DVC issues

```bash
# Check MinIO console
open http://localhost:9001   # Login: minioadmin/minioadmin

# Check DVC remote
uv run dvc remote list
uv run dvc push
```

---

## Frontend Issues

### Dashboard shows blank or errors

1. Check API is healthy: `curl http://localhost:8001/health`
2. Check Vite proxy target matches API port
3. Check browser console for CORS or network errors
4. Restart frontend: `cd frontend && npm run dev`

### Model selector shows duplicates

This was fixed — model selector now shows only champion (unique) models. If you see duplicates:

```bash
cd frontend && npm run build  # Rebuild
```

### Grafana embed not loading in dashboard

Check Grafana environment variables in `compose.yaml`:

```yaml
GF_SECURITY_ALLOW_EMBEDDING: "true"
GF_AUTH_ANONYMOUS_ENABLED: "true"
GF_AUTH_ANONYMOUS_ORG_ROLE: "Viewer"
```

---

## Development Issues

### Ruff/Mypy/Pytest failures

```bash
# Run all quality gates
uv run ruff check src/ tests/ scripts/ dags/    # Lint
uv run mypy src/ --ignore-missing-imports        # Type check
uv run pytest tests/ -q                          # Tests

# Auto-fix ruff issues
uv run ruff check --fix .
uv run ruff format .
```

### Import errors

- Domain layer must NOT import from infrastructure
- Always use absolute imports: `from src.domain.xxx import ...`
- Check `__init__.py` exports

### Docker build cache issues

```bash
docker compose build --no-cache api
docker compose up -d
```

---

## Port Reference

| Service | Internal Port | External Port |
|---------|--------------|---------------|
| API (FastAPI) | 8000 | **8001** |
| Frontend (Vite) | 5173 | **5174** |
| PostgreSQL | 5432 | **5433** |
| Redis | 6379 | **6380** |
| Kafka | 9092 | **9094** |
| Prometheus | 9090 | **9091** |
| Grafana | 3000 | **3001** |
| MinIO API | 9000 | **9000** |
| MinIO Console | 9001 | **9001** |
| Jaeger | 16686 | **16686** |
| MLflow | 5000 | **5000** |
| Airflow | 8080 | **8080** |
| gRPC | 50051 | **50051** |

---

## Getting Help

1. Check API logs: `docker compose logs -f api`
2. Check Airflow logs: `docker compose -f docker-compose.airflow.yaml logs -f`
3. Open Swagger docs: http://localhost:8001/docs
4. Open GitHub Issues: https://github.com/vtnguyen04/phoenix_ML/issues
