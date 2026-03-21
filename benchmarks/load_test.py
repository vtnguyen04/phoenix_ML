"""
Locust Load Test — Phoenix ML Platform.

Tests the /predict endpoint under various load patterns.

Usage:
    # Install: pip install locust
    # Run:    locust -f benchmarks/load_test.py --host http://localhost:8000

    # Headless mode (CI):
    locust -f benchmarks/load_test.py --host http://localhost:8000 \
           --headless -u 50 -r 10 --run-time 60s
"""

import random

from locust import HttpUser, between, task


class PredictUser(HttpUser):
    """Simulates a user making prediction requests."""

    wait_time = between(0.1, 0.5)

    @task(5)
    def predict_credit_risk(self) -> None:
        """Predict credit risk (30 features)."""
        features = [random.gauss(0, 1) for _ in range(30)]
        self.client.post(
            "/predict",
            json={
                "model_id": "credit-risk",
                "model_version": "v1",
                "features": features,
            },
        )

    @task(3)
    def predict_house_price(self) -> None:
        """Predict house price (8 features)."""
        features = [
            random.uniform(1, 10),  # MedInc
            random.uniform(10, 50),  # HouseAge
            random.uniform(3, 10),  # AveRooms
            random.uniform(0.5, 3),  # AveBedrms
            random.uniform(500, 5000),  # Population
            random.uniform(1, 6),  # AveOccup
            random.uniform(32, 42),  # Latitude
            random.uniform(-124, -114),  # Longitude
        ]
        self.client.post(
            "/predict",
            json={
                "model_id": "house-price",
                "model_version": "v1",
                "features": features,
            },
        )

    @task(2)
    def predict_fraud(self) -> None:
        """Predict fraud detection (12 features)."""
        features = [random.gauss(0, 1) for _ in range(12)]
        self.client.post(
            "/predict",
            json={
                "model_id": "fraud-detection",
                "model_version": "v1",
                "features": features,
            },
        )

    @task(1)
    def health_check(self) -> None:
        """Health check endpoint."""
        self.client.get("/health")

    @task(1)
    def get_model_info(self) -> None:
        """Get model info."""
        model = random.choice(["credit-risk", "house-price", "fraud-detection"])
        self.client.get(f"/models/{model}")


class BatchPredictUser(HttpUser):
    """Simulates batch prediction requests."""

    wait_time = between(1, 3)

    @task
    def batch_predict(self) -> None:
        """Batch prediction (10 items)."""
        batch_size = 10
        features_batch = [[random.gauss(0, 1) for _ in range(30)] for _ in range(batch_size)]
        self.client.post(
            "/predict/batch",
            json={
                "model_id": "credit-risk",
                "model_version": "v1",
                "batch": features_batch,
            },
        )
