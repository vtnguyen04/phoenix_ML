# System Design v3 - High-Performance, Low-Latency Architecture for Single Vehicle

## I. Overview

This document outlines a new, high-performance, low-latency architecture for the real-time monitoring dashboard, specifically tailored for a **single self-driving vehicle operating on a local network**. The design prioritizes minimal latency and direct communication to achieve the fastest possible data flow.

## II. Architecture

The entire system (Vehicle Agent, Go Backend, Frontend) is designed to run directly on the **Vehicle's Computer** (e.g., Jetson, Raspberry Pi). The operator accesses the dashboard from a standard web browser on a machine connected to the same local network as the vehicle.

```
┌───────────────────────────────────────────────────────────┐
│                    Operator's Machine                      │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                Web Browser                          │  │
│  │  - Accesses dashboard via http://<vehicle_ip>:8080  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
                           |
                           | HTTP (Frontend serving) / WebSocket (Telemetry)
                           |
┌───────────────────────────────────────────────────────────┐
│                    Vehicle's Computer                      │
│                                                           │
│  ┌──────────────────┐      ┌───────────────────────────┐  │
│  │  Vehicle Agent   │      │     Go Backend Service      │  │
│  │   (Python)      │      │ (single application)      │  │
│  │                 ├─────►│ - Serves static frontend  │  │
│  │ - Collects data │ UDP  │ - WebSocket server        │  │
│  │ - Sends Telemetry│      │ - UDP listener            │  │
│  └──────────────────┘      └───────────┬───────────────┘  │
│                                        │                  │
│                                        │                  │
│                                        ▼                  │
│                          ┌───────────────────────────┐  │
│                          │       Vite Frontend       │  │
│                          │  (built as static files)  │  │
│                          └───────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## III. Components

### 1. Vehicle Agent (Python)

*   **Responsibilities:**
    *   **Telemetry Data Collection:** Gathers real-time data from vehicle sensors (speed, steering, throttle, brake, GPS coordinates, heading, etc.). For development, this will be simulated data.
    *   **Direct UDP Transmission:** Sends the collected telemetry data as JSON packets directly to the Go Backend Service via UDP.
*   **Technology:** Python, `socket` module (for UDP).
*   **Communication:** UDP to `localhost:UDP_PORT` (e.g., `localhost:8082`).

### 2. Go Backend Service

*   **Responsibilities:**
    *   **Static File Server:** Serves the built static assets of the Vite React frontend (HTML, CSS, JavaScript) over HTTP on port `8080`.
    *   **WebSocket Server:** Provides a `/ws` endpoint for the frontend to establish a WebSocket connection. All connected clients receive real-time telemetry updates.
    *   **UDP Listener:** Binds to a specific UDP port (e.g., `8082`) and continuously listens for incoming telemetry data packets from the Vehicle Agent.
    *   **Telemetry Broadcasting:** Upon receiving a UDP telemetry packet, it immediately parses the JSON data and broadcasts it to all currently connected WebSocket clients.
*   **Technology:** Go, `net` package (for UDP), `gorilla/websocket` library.
*   **Communication:**
    *   HTTP for static files: `0.0.0.0:8080`
    *   WebSocket for real-time data: `ws://0.0.0.0:8080/ws`
    *   UDP listener for telemetry input: `0.0.0.0:8082`

### 3. Frontend (Vite + React)

*   **Responsibilities:**
    *   **WebSocket Client:** Establishes and maintains a WebSocket connection with the Go Backend Service (`ws://<vehicle_ip>:8080/ws`).
    *   **Real-time Dashboard Display:** Receives JSON telemetry data over the WebSocket and updates the UI elements (speed, steering, graphs, etc.) in real-time.
    *   **User Interface:** Provides a clean, responsive web interface for visualizing vehicle status.
*   **Technology:** React, Vite (for efficient static builds), native JavaScript `WebSocket` API.
*   **Communication:** WebSocket to `ws://localhost:8080/ws` (when accessed directly on vehicle, or `ws://<vehicle_ip>:8080/ws` from operator machine).

## IV. Data Flow Details

1.  **Telemetry Generation (Vehicle Agent):** The Python agent generates a JSON string representing the current vehicle telemetry state.
2.  **UDP Transmission (Vehicle Agent to Go Backend):** The agent creates a UDP packet containing the JSON telemetry string and sends it to the Go Backend Service's UDP listener port (`localhost:8082`).
3.  **UDP Reception & Parsing (Go Backend):** The Go Backend continuously reads UDP packets. Upon receipt, it parses the JSON string into a Go struct.
4.  **WebSocket Broadcasting (Go Backend to Frontend):** The parsed telemetry data is immediately re-serialized into a JSON string and sent to all currently active WebSocket connections at the `/ws` endpoint.
5.  **Frontend Display (Vite App):** The React frontend, connected via WebSocket, receives the JSON telemetry string, parses it, and updates the relevant components on the dashboard to reflect the latest data.

## V. Key Performance & Reliability Considerations

*   **Extremely Low Latency:**
    *   Direct UDP communication between agent and backend eliminates message broker overhead.
    *   In-memory parsing and direct WebSocket forwarding minimize processing delays.
    *   Frontend uses native WebSockets for efficient real-time updates.
    *   The entire stack running on the vehicle's computer removes network latency between components.
*   **Data Freshness:** The "fire-and-forget" nature of UDP prioritizes speed and freshness over guaranteed delivery. This is acceptable for a real-time monitoring dashboard where the latest data is always preferred over potentially stale but guaranteed data.
*   **Single Point of Failure:** While simplified, the single Go Backend Service is a single point of failure. This is a design trade-off for simplicity and performance in a dedicated single-vehicle context.
*   **Resource Efficiency:** Go and Python are chosen for their performance and efficiency on embedded systems like Raspberry Pi or Jetson.

## VI. Implementation Plan

1.  **Implement Go Backend Service:**
    *   Set up HTTP server for static files (`/`).
    *   Implement WebSocket endpoint (`/ws`).
    *   Implement UDP listener on port `8082`.
    *   Integrate UDP input to WebSocket broadcasting logic.
2.  **Implement Vehicle Agent Simulator:**
    *   Generate realistic telemetry data.
    *   Send data via UDP to `localhost:8082`.
3.  **Implement Frontend (Vite + React):**
    *   Connect to Go Backend via WebSocket (`ws://localhost:8080/ws`).
    *   Display received telemetry data in real-time.
    *   Develop core dashboard widgets (speed, steering, etc.).
4.  **Integrate and Test the complete system.**