import json
import socket
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

def main():
    # --- Configuration ---
    host = "127.0.0.1"
    port = 8080

    # --- Create Socket ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"Listening on {host}:{port}")

    # --- Receive Data ---
    data, addr = sock.recvfrom(65535)
    print(f"Received data from {addr}")

    # --- Decode Data ---
    payload = json.loads(data.decode())
    img_data = base64.b64decode(payload['image'])

    # --- Display Image ---
    img = Image.open(io.BytesIO(img_data))
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
