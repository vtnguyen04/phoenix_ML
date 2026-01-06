import random
import time

import requests


def send_requests(n: int = 50) -> None:
    url = "http://localhost:8001/predict"
    
    # Customer Profiles
    good_customer = "customer-good"
    bad_customer = "customer-bad"
    
    RANDOM_THRESHOLD = 0.5
    for i in range(n):
        entity = good_customer if random.random() > RANDOM_THRESHOLD else bad_customer
        
        payload = {
            "model_id": "credit-risk",
            # No model_version -> Dynamic Routing
            "entity_id": entity
        }
        
        HTTP_OK = 200
        try:
            resp = requests.post(url, json=payload, timeout=1)
            if resp.status_code == HTTP_OK:
                data = resp.json()
                print(
                    f"[{i+1}/{n}] Ver: {data['version']} | "
                    f"Result: {data['result']} | "
                    f"Conf: {data['confidence']:.2f}"
                )
            else:
                print(f"Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        time.sleep(0.1)

def check_metrics() -> None:
    HTTP_OK = 200
    try:
        resp = requests.get("http://localhost:8001/metrics", timeout=1)
        if resp.status_code == HTTP_OK:
            print("\n--- METRICS PREVIEW ---")
            for line in resp.text.split("\n"):
                if "prediction_count_total" in line and "#" not in line:
                    print(line)
                if "inference_latency_seconds_count" in line and "#" not in line:
                    print(line)
            print("-----------------------\n")
    except Exception as e:
        print(f"Metrics failed: {e}")

if __name__ == "__main__":
    print("Starting Traffic Simulation...")
    send_requests(20)
    check_metrics()
