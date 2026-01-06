# Deployment Guide: Production Orchestration

Phoenix ML is designed to be cloud-agnostic and container-first. This guide covers the deployment of the full stack using Docker Compose and provides a roadmap for Kubernetes migration.

## 1. Stack Architecture

The platform consists of seven interconnected services, ensuring high availability and separation of concerns.

| Service | Technology | Role |
| :--- | :--- | :--- |
| **api** | FastAPI + ONNX Runtime | Core Inference Engine |
| **frontend** | React + Vite | Control Plane Dashboard |
| **kafka** | Bitnami Kafka (KRaft) | Asynchronous Event Streaming |
| **redis** | Redis 7 | High-speed Online Feature Store |
| **db** | PostgreSQL 15 | Model Metadata & Audit Logs |
| **prometheus** | Prometheus 2 | Metrics Collection |
| **grafana** | Grafana 9 | Dashboard Visualization |

## 2. Infrastructure Requirements

### Recommended Specs (Local/Dev)
*   **CPU**: 4+ Cores (Inference is compute-intensive).
*   **RAM**: 8GB Minimum (16GB recommended for full stack).
*   **Storage**: 10GB SSD (For Kafka logs and Model artifacts).

### GPU Acceleration
To enable NVIDIA GPU acceleration for ONNX Runtime:
1.  Ensure `nvidia-docker2` is installed on the host.
2.  Update `Dockerfile` to use `onnxruntime-gpu`.
3.  Add the `deploy.resources.reservations.devices` section to `api` in `compose.yaml`.

## 3. Configuration & Security

### Secure Secrets Management
Do **not** store sensitive passwords in `compose.yaml` for production. Use a `.env` file (ignored by Git) or a secrets manager:
```bash
# Example .env entry
DATABASE_URL=postgresql+asyncpg://admin:SECURE_PASSWORD@db:5432/phoenix
```

### Persistence Strategy
The following volumes are defined to ensure zero data loss:
-   `models:/app/models`: Stores the downloaded ONNX artifacts.
-   `postgres_data:/var/lib/postgresql/data`: Stores metadata and logs.
-   `grafana_data:/var/lib/grafana`: Stores dashboard customizations.

## 4. Scaling & Production Roadmap

### Vertical vs. Horizontal Scaling
-   **API Horizontal Scaling**: Increase `api` replicas to handle more RPS.
    ```bash
    docker compose up -d --scale api=5
    ```
-   **Inference Vertical Scaling**: Move `api` to instances with higher memory bandwidth or GPU support.

### Kubernetes Migration (Production Target)
For enterprise scale, migrate the stack to Kubernetes:
1.  **Helm Charts**: Create charts for the `api` and `frontend`.
2.  **Managed Services**: Replace self-hosted Kafka/Postgres with managed services (e.g., Confluent Cloud, AWS RDS).
3.  **Ingress**: Use an Nginx Ingress Controller or Istio for advanced traffic routing and rate limiting.
4.  **HPA**: Configure Horizontal Pod Autoscaler based on the `inference_latency_seconds` metric from Prometheus.