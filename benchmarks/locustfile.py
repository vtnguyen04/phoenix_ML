"""
Locust Load Test for Phoenix ML Platform.

Usage:
    pip install locust
    locust -f benchmarks/locustfile.py --host http://localhost:8001

Web UI: http://localhost:8089
"""

from locust import HttpUser, between, task


class PhoenixMLUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict_with_entity(self) -> None:
        self.client.post(
            "/api/predict",
            json={
                "model_id": "credit-risk",
                "model_version": "v1",
                "entity_id": "customer-good",
            },
        )

    @task(3)
    def predict_with_features(self) -> None:
        self.client.post(
            "/api/predict",
            json={
                "model_id": "credit-risk",
                "model_version": "v1",
                "features": [0.5] * 30,
            },
        )

    @task(2)
    def health_check(self) -> None:
        self.client.get("/api/health")

    @task(1)
    def drift_scan(self) -> None:
        self.client.get("/api/monitoring/drift/credit-risk")

    @task(1)
    def get_model_info(self) -> None:
        self.client.get("/api/models/credit-risk")
