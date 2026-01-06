# API Reference: Phoenix ML Platform

The Phoenix ML API provides a high-performance REST interface for real-time inference and system monitoring.

## Base URL
-   **Local**: `http://localhost:8000`
-   **Production**: Defined by your ingress controller (e.g., `https://ml.phoenix.com`)

---

## 1. Real-time Inference

### `POST /predict`
Executes a machine learning model prediction. The system will automatically route to the best model version unless a specific version is requested.

#### Request Body (JSON)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | string | Yes | The name of the model group (e.g., `credit-risk`). |
| `model_version` | string | No | Specific version (e.g., `v1`). Defaults to dynamic A/B routing. |
| `entity_id` | string | No | ID to fetch features from the Feature Store (Redis). |
| `features` | array[float] | No | Raw feature vector. Overrides Feature Store if provided. |

#### Example Request
```json
{
  "model_id": "credit-risk",
  "entity_id": "customer-good"
}
```

#### Response (200 OK)
| Field | Type | Description |
| :--- | :--- | :--- |
| `model_id` | string | The model ID used for inference. |
| `version` | string | The assigned version (determined by router). |
| `result` | integer | The prediction class (e.g., `0` or `1`). |
| `confidence` | float | The confidence score (0.0 to 1.0). |
| `latency_ms` | float | Server-side execution time in milliseconds. |

---

## 2. Model Monitoring

### `GET /monitoring/drift/{model_id}`
Triggers an on-demand statistical drift check for a specific model group.

#### Path Parameters
-   `model_id` (string, required): The ID of the model to analyze.

#### Response (200 OK)
```json
{
  "feature_name": "feature_0",
  "drift_detected": true,
  "p_value": 0.00001,
  "statistic": 0.875,
  "threshold": 0.05
}
```

---

## 3. System Observability

### `GET /metrics`
Exposes system-wide Prometheus metrics. Used by the Prometheus scraper.

### `GET /health`
Liveness and readiness probe for the service.

---

## Error Handling

All errors return a standard JSON object:
```json
{
  "detail": "Error message description"
}
```

### Common HTTP Status Codes
| Code | Name | Scenario |
| :--- | :--- | :--- |
| **404** | Not Found | Requested model or version is not registered or active. |
| **422** | Validation Error | Input data does not match the expected schema (e.g., non-numeric features). |
| **500** | Internal Error | Inference engine crash or database connection failure. |
| **503** | Service Unavailable | Circuit breaker is open due to repeated upstream failures. |