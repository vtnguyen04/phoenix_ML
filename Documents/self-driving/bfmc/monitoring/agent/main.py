import asyncio
import json
import random
import socket
from datetime import datetime, timezone

async def run():
    udp_ip = "127.0.0.1"
    udp_port = 8082 # Must match the Go backend's UDP listener port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

    while True:
        telemetry_data = {
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "VehicleID": "test-vehicle", # Hardcoded for now
            "Speed": random.uniform(0, 120),
            "SteeringAngle": random.uniform(-30, 30),
            "Throttle": random.uniform(0, 1),
            "Brake": random.uniform(0, 1),
            "Gear": random.choice([1, 2, 3, 4, 5, 6]),
            "Latitude": random.uniform(34.0, 36.0),
            "Longitude": random.uniform(-118.0, -116.0),
            "Heading": random.uniform(0, 360),
            "Metadata": {},
        }

        message = json.dumps(telemetry_data).encode('utf-8')
        sock.sendto(message, (udp_ip, udp_port))
        print(f"Sent UDP: {telemetry_data}")

        await asyncio.sleep(0.1) # Send at 10 Hz

if __name__ == '__main__':
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Vehicle agent simulator stopped.")
