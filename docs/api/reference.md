# API Reference: Phoenix ML Platform

Base URL: `http://localhost:8001` (Docker) · `http://localhost:8000` (local dev)

> **Note**: All endpoints have NO `/api/` prefix. Routes are mounted directly at root.

---

## 1. Health & Status

### `GET /health`
Service health check.

**Response (200)**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

## 2. Inference

### `POST /predict`
Execute ML model prediction with automatic model routing.

**Request Body**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model group name (e.g., `credit-risk`, `house-price`, `fraud-detection`, `image-class`) |
| `model_version` | string | No | Specific version (e.g., `v1`). Defaults to A/B routing |
| `entity_id` | string | No | Entity ID to fetch features from Redis |
| `features` | array[float] | No | Raw feature vector (overrides feature store) |

**Response (200)**
```json
{
  "prediction_id": "a1b2c3d4-...",
  "model_id": "credit-risk",
  "version": "v1",
  "result": 1,
  "confidence": {"value": 0.87},
  "latency_ms": 12.3
}
```

### `POST /predict/batch`
Batch prediction — process multiple inputs concurrently.

**Request Body**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Target model |
| `batch` | array[array[float]] | Yes | List of feature vectors |
| `model_version` | string | No | Specific version |
| `entity_ids` | array[string] | No | Entity IDs for feature store lookup |

**Response (200)**
```json
{
  "predictions": [...],
  "total": 10,
  "successful": 9,
  "errors": [{"index": 3, "error": "..."}],
  "batch_latency_ms": 45.2
}
```

---

## 3. Model Management

### `GET /models/{model_id}`
Get model metadata and status.

**Response (200)**
```json
{
  "model_id": "credit-risk",
  "version": "v1",
  "status": "active",
  "metadata": {
    "role": "champion",
    "model_type": "GradientBoostingClassifier",
    "n_features": 30,
    "metrics": {"accuracy": 0.785, "f1_score": 0.854}
  }
}
```

### `POST /models/register`
Register a new model version in the registry.

**Request Body**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model identifier |
| `version` | string | Yes | Version tag (e.g., `v2`) |
| `uri` | string | Yes | Model artifact URI |
| `framework` | string | No | Default: `onnx` |
| `stage` | string | No | Default: `challenger` |
| `metadata` | object | No | Additional metadata |
| `metrics` | object | No | Training metrics |

### `POST /models/rollback`
Archive all active challenger models, keeping champion serving.

**Request Body**
```json
{"model_id": "credit-risk"}
```

---

## 4. Monitoring

### `GET /monitoring/drift/{model_id}`
Trigger on-demand drift detection using KS statistics.

**Response (200)**
```json
{
  "feature_name": "feature_0",
  "drift_detected": true,
  "method": "ks_2samp",
  "p_value": 0.00001,
  "statistic": 0.875,
  "threshold": 0.05,
  "sample_size": 100
}
```

### `GET /monitoring/reports/{model_id}?limit=10`
Get historical drift reports.

### `GET /monitoring/performance/{model_id}`
Get aggregated prediction performance metrics.

---

## 5. Feedback Loop

### `POST /feedback`
Submit ground-truth feedback for online evaluation.

**Request Body**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prediction_id` | string | Yes | ID from predict response |
| `ground_truth` | int/float/string | Yes | Actual label or value |

---

## 6. Feature Store

### `GET /features/{entity_id}`
Retrieve features for an entity from the online store (Redis).

**Response (200)**
```json
{
  "entity_id": "customer-good",
  "features": {"income": 50000, "age": 35, "credit_score": 720}
}
```

### `POST /features/ingest`
Ingest feature data into the store.

### `GET /features/reference`
Get reference feature distributions for drift comparison.

---

## 7. Observability

### `GET /metrics`
Prometheus metrics endpoint (scraped automatically by Prometheus).

Key metrics exposed:
- `prediction_count_total` — Total predictions by model/version
- `inference_latency_seconds` — Histogram of inference latency
- `model_confidence` — Gauge of prediction confidence
- `feature_drift_score` — Drift score per feature
- `drift_detected_events_total` — Counter of drift events

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