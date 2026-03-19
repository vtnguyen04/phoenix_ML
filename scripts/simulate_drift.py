import os
import time

import numpy as np
import requests

DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "credit-risk")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
N_FEATURES = int(os.environ.get("N_FEATURES", "30"))


def send_drifted_traffic(n: int = 50, model_id: str = DEFAULT_MODEL_ID) -> None:
    url = f"{API_URL}/predict"

    print(f"🌊 Sending {n} DRIFTED requests for model '{model_id}'...")

    for i in range(n):
        # Generate drifted features (feature 0 is shifted)
        features = [float(np.random.normal(0, 1)) for _ in range(N_FEATURES)]
        features[0] = float(np.random.normal(5, 2))  # Drift on feature 0

        payload = {
            "model_id": model_id,
            "features": features,
        }

        try:
            requests.post(url, json=payload, timeout=0.5)
        except Exception:
            pass  # Ignore errors, we just want to fill the log

        if i % 10 == 0:
            print(f"Sent {i}...")
        time.sleep(0.05)


def check_drift_metrics() -> None:
    HTTP_OK = 200
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=1)
        if resp.status_code == HTTP_OK:
            print("\n--- DRIFT METRICS STATUS ---")
            found = False
            for line in resp.text.split("\n"):
                if "feature_drift_score" in line and "#" not in line:
                    print(f"📉 {line}")
                    found = True
                if "drift_detected_events_total" in line and "#" not in line:
                    print(f"🚨 {line}")
                    found = True

            if not found:
                print("No drift metrics recorded yet.")
            print("----------------------------\n")
    except Exception as e:
        print(f"Metrics failed: {e}")


if __name__ == "__main__":
    send_drifted_traffic(100)  # Send enough data to trigger threshold (MIN_POINTS=5)
    print("Waiting for Monitoring Loop (5s)...")
    time.sleep(6)
    check_drift_metrics()
