import time

import numpy as np
import requests


def send_drifted_traffic(n: int = 50) -> None:
    url = "http://localhost:8001/predict"
    
    print(f"🌊 Sending {n} DRIFTED requests...")
    
    for i in range(n):
        # Generate Drifted Data for Feature 0 (Income)
        # Normal traffic: mean=0, std=1
        # Drifted traffic: mean=5, std=2
        f0 = float(np.random.normal(5, 2)) 
        f1 = float(np.random.normal(0, 1))
        f2 = float(np.random.normal(0, 1))
        f3 = float(np.random.normal(0, 1))
        
        payload = {
            "model_id": "credit-risk",
            # "model_version": "v1", # Let router decide or stick to v1
            "features": [f0, f1, f2, f3] # Override feature store
        }
        
        try:
            requests.post(url, json=payload, timeout=0.5)
        except Exception:
            pass # Ignore errors, we just want to fill the log
            
        if i % 10 == 0:
            print(f"Sent {i}...")
        time.sleep(0.05)

def check_drift_metrics() -> None:
    HTTP_OK = 200
    try:
        resp = requests.get("http://localhost:8001/metrics", timeout=1)
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
    send_drifted_traffic(100) # Send enough data to trigger threshold (MIN_POINTS=5)
    print("Waiting for Monitoring Loop (5s)...")
    time.sleep(6) 
    check_drift_metrics()
