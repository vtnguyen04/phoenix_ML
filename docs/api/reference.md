# API Reference: Phoenix ML Platform

Base URL: `http://localhost:8001`

---

## 1. Health & Status

### `GET /api/health`
Service health check with dependency status.

**Response (200)**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "kafka_connected": true,
  "database_connected": true
}
```

---

## 2. Inference

### `POST /api/predict`
Execute ML model prediction with automatic model routing.

**Request Body**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model group name (e.g., `credit-risk`) |
| `model_version` | string | No | Specific version. Defaults to A/B routing |
| `entity_id` | string | No | Entity ID to fetch features from Redis |
| `features` | array[float] | No | Raw feature vector (overrides feature store) |

**Response (200)**
```json
{
  "model_id": "credit-risk",
  "version": "v1",
  "result": 1,
  "confidence": 0.87,
  "latency_ms": 12.3
}
```

---

## 3. Model Management

### `GET /api/models/{model_id}`
Get model metadata and performance metrics.

**Response (200)**
```json
{
  "model_id": "credit-risk",
  "model_type": "GradientBoostingClassifier",
  "dataset": "german-credit-openml",
  "n_features": 30,
  "train_samples": 800,
  "metrics": {
    "accuracy": 0.785,
    "f1_score": 0.854,
    "precision": 0.813,
    "recall": 0.900
  }
}
```

---

## 4. Monitoring

### `GET /api/monitoring/drift/{model_id}`
Trigger on-demand drift detection using statistical tests.

**Response (200)**
```json
{
  "feature_name": "credit_amount",
  "drift_detected": true,
  "method": "ks_2samp",
  "p_value": 0.00001,
  "statistic": 0.875,
  "threshold": 0.05,
  "sample_size": 100
}
```

### `GET /api/monitoring/reports/{model_id}`
Get historical drift reports.

### `GET /api/monitoring/performance/{model_id}`
Get model performance metrics over time.

---

## 5. Feature Store

### `GET /api/features/{entity_id}`
Retrieve features for an entity from the online feature store (Redis).

**Response (200)**
```json
{
  "entity_id": "customer-good",
  "features": [1.2, 0.5, 3.4, ...],
  "source": "redis"
}
```

---

## 6. Observability

### `GET /metrics`
Prometheus metrics endpoint (scraped automatically).

---

## Error Handling

```json
{"detail": "Error message description"}
```

| Code | Scenario |
|------|----------|
| 404 | Model or version not found |
| 422 | Invalid request schema |
| 500 | Inference engine or database failure |
| 503 | Circuit breaker open |

---
*Updated March 2026*