# API Reference: Phoenix ML Platform

Base URL: `http://localhost:8001` (Docker) · `http://localhost:8000` (local dev)

## Overview

Phoenix ML exposes a RESTful API via FastAPI and gRPC (port 50051). All endpoints return JSON.

---

## Health Check

### `GET /health`

Checks that the API server is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## Inference

### `POST /predict`

Performs a single prediction.

**Request Body:**
```json
{
  "model_id": "credit-risk",
  "model_version": "v1",        // Optional, defaults to champion
  "entity_id": "customer_42",   // Optional, fetches features from Redis
  "features": [0.5, 1.2, 0.8, 3.4, 0.1, 2.5, 1.0]  // Optional if entity_id provided
}
```

**Response (200 OK):**
```json
{
  "model_id": "credit-risk",
  "model_version": "v1",
  "result": 1,
  "confidence": 0.87,
  "latency_ms": 2.34
}
```

**Errors:**

| Status | Reason |
|--------|--------|
| 404 | Model not found |
| 422 | Invalid features / missing required fields |
| 503 | Circuit breaker OPEN (model overloaded) |

**Processing flow:**

1. Router receives request → creates `PredictCommand`
2. `PredictHandler.handle(command)`:
   - Resolves model (champion or specific version)
   - Gets features: from entity_id (Redis) or raw features in request
   - Runs inference via `InferenceEngine.predict(model, features)`
3. Background tasks:
   - Logs prediction to PostgreSQL
   - Publishes event to Kafka topic `inference-events`
4. Returns `PredictionResponse`

---

### `POST /predict/batch`

Performs batch prediction for multiple inputs at once.

**Request Body:**
```json
{
  "model_id": "credit-risk",
  "model_version": "v1",
  "batch": [
    [0.5, 1.2, 0.8, 3.4, 0.1, 2.5, 1.0],
    [0.3, 0.9, 1.1, 2.8, 0.2, 1.5, 0.8],
    [0.7, 1.5, 0.6, 4.1, 0.3, 3.0, 1.2]
  ],
  "entity_ids": ["c_01", "c_02", "c_03"]  // Optional
}
```

**Response (200 OK):**
```json
{
  "predictions": [
    {"result": 1, "confidence": 0.87, "latency_ms": 1.5},
    {"result": 0, "confidence": 0.92, "latency_ms": 1.3},
    {"result": 1, "confidence": 0.78, "latency_ms": 1.4}
  ],
  "model_id": "credit-risk",
  "total_latency_ms": 4.2
}
```

---

## Feedback

### `POST /feedback`

Submits ground truth for a completed prediction (used for model evaluation).

**Request Body:**
```json
{
  "prediction_id": "pred_abc123",
  "ground_truth": 1
}
```

**Response (200 OK):**
```json
{
  "status": "feedback_recorded",
  "prediction_id": "pred_abc123"
}
```

---

## Model Management

### `GET /models`

Lists all active models in the registry.

**Response (200 OK):**
```json
{
  "models": [
    {
      "id": "credit-risk",
      "version": "v1",
      "framework": "onnx",
      "stage": "champion",
      "is_active": true,
      "metadata": {
        "metrics": {"accuracy": 0.95, "f1_score": 0.93},
        "features": ["income", "age", "credit_score", "..."]
      },
      "created_at": "2026-03-19T10:30:00Z"
    },
    {
      "id": "fraud-detection",
      "version": "v1",
      "framework": "onnx",
      "stage": "champion",
      "is_active": true,
      "metadata": {
        "metrics": {"accuracy": 0.97, "f1_score": 0.96}
      }
    }
  ]
}
```

### `GET /models/{model_id}`

Returns detailed information for a single model (champion version).

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `model_id` | path | Model identifier (e.g., `credit-risk`) |

**Response (200 OK):**
```json
{
  "id": "credit-risk",
  "version": "v1",
  "uri": "local:///models/credit_risk/v1/model.onnx",
  "framework": "onnx",
  "stage": "champion",
  "is_active": true,
  "metadata": {
    "role": "champion",
    "metrics": {
      "accuracy": 0.95,
      "f1_score": 0.93,
      "precision": 0.94,
      "recall": 0.92
    },
    "features": ["income", "age", "credit_score", "debt_ratio", "num_accounts", "credit_history", "employment_years"]
  },
  "created_at": "2026-03-19T10:30:00Z"
}
```

### `POST /models/register`

Registers a new model version.

**Request Body:**
```json
{
  "model_id": "credit-risk",
  "version": "v2",
  "uri": "s3://models/credit-risk/v2/model.onnx",
  "framework": "onnx",
  "metadata": {
    "metrics": {"accuracy": 0.96, "f1_score": 0.95},
    "features": ["income", "age", "credit_score"]
  }
}
```

**Response (201 Created):**
```json
{
  "status": "registered",
  "model_id": "credit-risk",
  "version": "v2",
  "stage": "challenger"
}
```

### `POST /models/rollback`

Archives all challenger versions and keeps the current champion.

**Request Body:**
```json
{
  "model_id": "credit-risk"
}
```

**Response (200 OK):**
```json
{
  "status": "rolled_back",
  "model_id": "credit-risk",
  "champion_version": "v1",
  "archived_versions": ["v2", "v3"]
}
```

---

## Monitoring

### `GET /monitoring/drift/{model_id}`

Runs drift detection for a model and returns real-time results.

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `model_id` | path | Model identifier |

**Response (200 OK):**
```json
{
  "model_id": "credit-risk",
  "drift_score": 0.12,
  "is_drifted": false,
  "method": "ks",
  "threshold": 0.3,
  "feature_name": "feature_0",
  "timestamp": "2026-03-22T10:30:00Z"
}
```

**Drift Methods:**

| Method | Algorithm | When to use |
|--------|-----------|-------------|
| `ks` | Kolmogorov-Smirnov test | Continuous features, general purpose |
| `psi` | Population Stability Index | Binned distributions, fraud detection |
| `chi2` | Chi-squared test | Categorical features |

### `GET /monitoring/reports/{model_id}`

Returns historical drift reports.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 10 | Maximum number of reports |

**Response (200 OK):**
```json
{
  "model_id": "credit-risk",
  "reports": [
    {
      "drift_score": 0.05,
      "is_drifted": false,
      "method": "ks",
      "timestamp": "2026-03-22T10:30:00Z"
    },
    {
      "drift_score": 0.42,
      "is_drifted": true,
      "method": "ks",
      "timestamp": "2026-03-22T10:00:00Z"
    }
  ]
}
```

### `GET /monitoring/performance/{model_id}`

Returns performance metrics for a model (computed from logs with ground truth).

**Response (200 OK):**
```json
{
  "model_id": "credit-risk",
  "metrics": {
    "accuracy": 0.93,
    "f1_score": 0.91,
    "precision": 0.92,
    "recall": 0.90
  },
  "num_predictions": 1250,
  "num_with_ground_truth": 875
}
```

---

## Data Pipeline

### `POST /data/ingest`

Runs the data pipeline: load → validate → clean → store.

**Request Body:**
```json
{
  "source_path": "data/credit_risk/dataset.csv",
  "target_column": "target",
  "output_path": "data/credit_risk/cleaned.csv",
  "feature_ranges": {"age": [18, 75], "income": [0, 500000]}
}
```

### `POST /data/validate`

Validates data quality without ingesting.

**Request Body:**
```json
{
  "source_path": "data/credit_risk/dataset.csv",
  "target_column": "target"
}
```

### `POST /data/export-training`

Exports fresh labeled data from prediction logs for self-healing retrain.

**Request Body:**
```json
{
  "model_id": "credit-risk",
  "min_samples": 500,
  "include_baseline": true,
  "max_fresh_samples": 10000
}
```

**Response (200 OK):**
```json
{
  "export_path": "data/credit_risk/retrain_1774283667.csv",
  "total_samples": 1042,
  "fresh_samples": 42,
  "baseline_samples": 1000,
  "model_id": "credit-risk"
}
```

**How it works:**
1. Queries `prediction_logs` WHERE `ground_truth IS NOT NULL` (labeled by humans via `/feedback`)
2. Loads baseline training CSV from model config
3. Aligns column names between fresh logs and baseline
4. Merges: `baseline + fresh → combined dataset`
5. Writes timestamped CSV for the retrain pipeline

---

## Retrain Trigger

### `POST /models/{model_id}/retrain`

Manually triggers model retraining via Airflow self-healing DAG.

**Response (200 OK):**
```json
{
  "model_id": "credit-risk",
  "status": "triggered",
  "dag_run_id": "manual__2026-03-23T15:30:00",
  "message": "Retrain pipeline triggered for credit-risk"
}
```

**Errors:**

| Status | Reason |
|--------|--------|
| 502 | Airflow not available or DAG not found |

---

### `GET /features/{entity_id}`

Retrieves features for an entity from the online store.

**Response (200 OK):**
```json
{
  "entity_id": "customer_42",
  "features": {
    "income": 55000.0,
    "age": 32.0,
    "credit_score": 720.0,
    "debt_ratio": 0.25
  }
}
```

### `POST /features/{entity_id}`

Adds or updates features for an entity.

**Request Body:**
```json
{
  "features": {
    "income": 60000.0,
    "age": 33.0,
    "credit_score": 740.0
  }
}
```

---

## gRPC API

Port: `50051`

### Service Definition (Proto)

```protobuf
service InferenceService {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

message PredictRequest {
  string model_id = 1;
  string model_version = 2;
  repeated float features = 3;
}

message PredictResponse {
  float result = 1;
  float confidence = 2;
  float latency_ms = 3;
  string model_id = 4;
}
```

### Python Client Example

```python
import grpc
from phoenix_ml.infrastructure.grpc.proto import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

response = stub.Predict(inference_pb2.PredictRequest(
    model_id="credit-risk",
    features=[0.5, 1.2, 0.8, 3.4, 0.1, 2.5, 1.0]
))
print(f"Result: {response.result}, Confidence: {response.confidence}")
```

---

## Prometheus Metrics

Endpoint: `GET /metrics`

| Metric | Type | Description |
|--------|------|-------------|
| `phoenix_prediction_count` | Counter | Total predictions served |
| `phoenix_inference_latency_ms` | Histogram | Inference latency (ms) |
| `phoenix_model_confidence` | Histogram | Prediction confidence scores |
| `phoenix_drift_score` | Gauge | Current drift score per model |
| `phoenix_drift_detected_total` | Counter | Total drift detections |
| `phoenix_model_accuracy` | Gauge | Model accuracy (from feedback) |
| `phoenix_model_f1_score` | Gauge | Model F1 score |
| `phoenix_model_rmse` | Gauge | Model RMSE (regression) |
| `phoenix_model_mae` | Gauge | Model MAE (regression) |
| `phoenix_model_r2` | Gauge | Model R² (regression) |
| `phoenix_model_primary_metric` | Gauge | Primary metric per model |

---
*Document Status: v4.0 — Updated March 2026*