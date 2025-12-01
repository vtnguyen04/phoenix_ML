## II. SYSTEM DESIGN

### 2.1 Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WEB CLIENT TIER                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  React SPA + TypeScript                                     │    │
│  │  - Dashboard Components (Charts, Cards, Video Player)       │    │
│  │  - WebSocket Client (Socket.io / native WebSocket)          │    │
│  │  - State Management (Redux/Zustand + RTK Query)             │    │
│  │  - Visualization (Recharts, D3.js, Three.js for 3D)         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              ↕ HTTPS/WSS                             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY / BFF TIER                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  API Gateway (Kong / Nginx + WebSocket Proxy)               │    │
│  │  - Authentication & Authorization                           │    │
│  │  - Rate Limiting & Request Validation                       │    │
│  │  - WebSocket Connection Management                          │    │
│  │  - Load Balancing                                           │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION SERVICES TIER                       │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ WebSocket Server │  │   REST API       │  │  Auth Service    │  │
│  │  (Node.js)       │  │  (FastAPI/Go)    │  │   (Go/Node.js)   │  │
│  │                  │  │                  │  │                  │  │
│  │ - Broadcast data │  │ - CRUD ops       │  │ - JWT issuing    │  │
│  │ - Room mgmt      │  │ - Query data     │  │ - User mgmt      │  │
│  │ - Client state   │  │ - Config mgmt    │  │ - RBAC           │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│           │                     │                      │             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Telemetry Svc    │  │ Video Stream Svc │  │ Control Svc      │  │
│  │  (Go/Python)     │  │  (Go + FFmpeg)   │  │   (Go/Rust)      │  │
│  │                  │  │                  │  │                  │  │
│  │ - Data ingestion │  │ - Transcoding    │  │ - Cmd validation │  │
│  │ - Processing     │  │ - Adaptive BR    │  │ - Safety checks  │  │
│  │ - Enrichment     │  │ - Multi-protocol │  │ - Cmd queueing   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│           │                     │                      │             │
└───────────┼─────────────────────┼──────────────────────┼─────────────┘
            │                     │                      │
┌───────────┼─────────────────────┼──────────────────────┼─────────────┐
│           ↓                     ↓                      ↓              │
│                       MESSAGE BUS / STREAMING TIER                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Apache Kafka / NATS / Redis Streams                         │    │
│  │                                                              │    │
│  │  Topics:                                                     │    │
│  │  - vehicle.telemetry.{vehicle_id}                           │    │
│  │  - vehicle.video.{vehicle_id}.{camera_id}                   │    │
│  │  - vehicle.control.commands.{vehicle_id}                    │    │
│  │  - vehicle.planning.{vehicle_id}                            │    │
│  │  - vehicle.events.{vehicle_id}                              │    │
│  │  - system.alerts                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└───────────┼─────────────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────────────┐
│           ↓                                                           │
│                      DATA PROCESSING TIER                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Stream Processor │  │  Data Archiver   │  │  Alert Engine    │  │
│  │ (Kafka Streams/  │  │   (Go/Python)    │  │   (Go)           │  │
│  │  Flink)          │  │                  │  │                  │  │
│  │                  │  │ - S3/MinIO store │  │ - Rule engine    │  │
│  │ - Aggregation    │  │ - Compression    │  │ - Notification   │  │
│  │ - Windowing      │  │ - Indexing       │  │ - Escalation     │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                         STORAGE TIER                                 │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ TimescaleDB /    │  │  PostgreSQL      │  │  Object Storage  │  │
│  │ InfluxDB         │  │                  │  │  (S3/MinIO)      │  │
│  │                  │  │ - Users, roles   │  │                  │  │
│  │ - Telemetry TS   │  │ - Configs        │  │ - Video files    │  │
│  │ - Metrics        │  │ - Sessions       │  │ - Logs archive   │  │
│  │ - Aggregates     │  │ - Annotations    │  │ - Snapshots      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │  Redis           │  │  Elasticsearch   │                         │
│  │                  │  │   (optional)     │                         │
│  │ - Session cache  │  │                  │                         │
│  │ - Real-time data │  │ - Log indexing   │                         │
│  │ - Pub/Sub        │  │ - Full-text srch │                         │
│  └──────────────────┘  └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY & MONITORING                         │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Prometheus +     │  │  Grafana         │  │ Jaeger/Tempo     │  │
│  │ Node Exporter    │  │                  │  │                  │  │
│  │                  │  │ - Dashboards     │  │ - Distributed    │  │
│  │ - Metrics scrape │  │ - Alerting       │  │   tracing        │  │
│  │ - Alertmanager   │  │ - Visualization  │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Loki / ELK Stack (Logs aggregation)                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↕
┌─────────────────────────────────────────────────────────────────────┐
│                         VEHICLE EDGE TIER                            │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Vehicle Agent (Python/C++ on Jetson/RPi)                    │    │
│  │                                                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │    │
│  │  │ Data Collector│ │ Video Encoder│  │ Cmd Receiver │      │    │
│  │  │              │  │              │  │              │      │    │
│  │  │ - Sensor poll│  │ - H.264/265  │  │ - Validation │      │    │
│  │  │ - Aggregation│  │ - Streaming  │  │ - Execution  │      │    │
│  │  │ - Buffering  │  │              │  │ - ACK        │      │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │    │
│  │                                                              │    │
│  │  Communication: Kafka client / gRPC / WebSocket / MQTT       │    │
│  │  Fallback: Local SD card logging khi mất kết nối            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                   ↕                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Vehicle Modules (ROS2 / Custom Stack)                       │    │
│  │  - Perception | Planning | Control | Localization            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Chi Tiết

#### 2.2.1 Telemetry Flow (Vehicle → Web)

```
[Vehicle Sensors] → [Data Collector on Vehicle]
                          ↓ (aggregate 10-50Hz)
                    [Protobuf/JSON payload]
                          ↓
                    [Kafka Producer / gRPC stream]
                          ↓
              [Message Bus: vehicle.telemetry.{id}]
                          ↓
         ┌────────────────┼────────────────┐
         ↓                ↓                ↓
  [Telemetry Svc]  [Stream Processor]  [Data Archiver]
  - Validation     - Windowing         - TimescaleDB
  - Enrichment     - Aggregation       - S3 backup
         ↓                ↓
  [Redis Cache]    [Alert Engine]
  (latest state)   (threshold check)
         ↓
  [WebSocket Server]
  - Fan-out to clients
  - Rate limiting per client
         ↓
  [Web Client: Dashboard Update]
```

**Latency Budget:**
- Vehicle sensor read: ~10ms
- Kafka publish: ~5-10ms
- Processing + enrichment: ~20ms
- WebSocket broadcast: ~10-20ms
- Network (WiFi/4G): ~20-50ms
- **Total: 65-110ms** ✓ (target <100ms)

#### 2.2.2 Video Flow (Vehicle → Web)

```
[Camera] → [Video Encoder on Vehicle]
                ↓ (H.264, 720p@30fps or adaptive)
          [UDP/RTP stream or Kafka]
                ↓
      [Video Stream Service]
      - Transcoding (if needed)
      - Adaptive bitrate
      - Add overlays (detection boxes)
                ↓
      [WebRTC / HLS / RTMP / WebSocket]
                ↓
      [Web Client: Video Player]
      - Low-latency playback
      - Canvas overlay rendering
```

**Options & Trade-offs:**

| Protocol | Latency | Quality | Complexity | Use Case |
|----------|---------|---------|------------|----------|
| **WebRTC** | 100-300ms | High | High | Best cho realtime, peer-to-peer |
| **HLS** | 2-10s | High | Low | Good cho playback quality, not realtime |
| **RTMP** | 500ms-2s | High | Medium | Good balance |
| **WebSocket + MJPEG** | 200-500ms | Medium | Low | Simple, easy debug |

**Recommendation**: WebRTC cho production, WebSocket+MJPEG cho prototyping.

#### 2.2.3 Control Flow (Web → Vehicle)

```
[User Action on Web] → [WebSocket Server]
                              ↓
                    [Control Service]
                    - Authentication check
                    - Authorization check (RBAC)
                    - Command validation
                    - Rate limiting
                    - Duplicate detection (sequence #)
                    - Audit logging
                              ↓
              [Kafka: vehicle.control.commands.{id}]
              (persistent, at-least-once)
                              ↓
            [Vehicle Command Receiver]
            - Deduplication
            - Safety validation
            - Execution
            - ACK back to backend
                              ↓
            [Vehicle Control Module]
            (steering, throttle, brake)
                              ↓
              [ACK Flow Back]
              → [Kafka] → [Control Svc] → [WebSocket] → [Web]
```

**Latency Budget:**
- Web action → WebSocket: ~5ms
- Control service processing: ~10-15ms
- Kafka publish: ~5-10ms
- Network to vehicle: ~20-50ms
- Vehicle validation + exec: ~10-20ms
- **Total: 50-100ms** ✓ (target <50ms optimistic, <100ms realistic)

**Safety Mechanisms:**
1. **Sequence numbering**: Mỗi command có unique ID + timestamp
2. **Idempotency**: Vehicle ignore duplicate commands (same seq #)
3. **Timeout**: Command expire sau 1s nếu không execute
4. **Heartbeat**: Vehicle gửi heartbeat mỗi 100ms, backend detect timeout
5. **Emergency override**: Emergency stop bypass tất cả queue, priority cao nhất

### 2.3 Technology Stack Chi Tiết

#### 2.3.1 Message Bus Options

| Technology | Pros | Cons | Verdict |
|------------|------|------|---------|
| **Apache Kafka** | - High throughput<br>- Persistent storage<br>- Strong ordering guarantee<br>- Mature ecosystem | - Heavy resource<br>- Complex setup<br>- Higher latency (~5-10ms) | **Recommended** nếu cần persistent replay và multi-consumer |
| **NATS / NATS Streaming** | - Very low latency (<1ms)<br>- Lightweight<br>- Simple deployment<br>- Good for realtime | - Limited persistence (JetStream add-on)<br>- Smaller ecosystem | **Recommended** nếu prioritize latency và đơn giản |
| **Redis Streams** | - Very low latency<br>- Simple<br>- Multi-purpose (cache + stream) | - Limited scalability vs Kafka<br>- Less mature for streaming | **Viable** cho prototype/small scale |
| **MQTT** | - IoT-optimized<br>- Low bandwidth<br>- Offline support | - QoS 2 có latency cao<br>- Not designed for high throughput | Good cho vehicle communication, not for backend |

**Final Recommendation**: 
- **NATS JetStream** cho production (best latency + good enough persistence)
- **Kafka** nếu cần compliance, audit, long-term retention
- **MQTT** cho vehicle ↔ backend communication (fallback mechanism)

#### 2.3.2 Time-Series Database

| Database | Pros | Cons | Verdict |
|----------|------|------|---------|
| **TimescaleDB** | - PostgreSQL extension<br>- SQL familiar<br>- ACID<br>- Rich query | - PostgreSQL overhead<br>- Write performance moderate | **Recommended** (best balance) |
| **InfluxDB** | - Purpose-built for TS<br>- High write throughput<br>- InfluxQL/Flux | - Separate ecosystem<br>- Clustering complex (OSS) | Viable alternative |
| **Prometheus** | - Great for metrics<br>- Pull model<br>- Grafana integration | - Not for high-cardinality data<br>- Limited retention | Use for system metrics only |

**Recommendation**: **TimescaleDB** (primary) + **Prometheus** (system metrics)

#### 2.3.3 Frontend Stack

```
- Framework: React 18+ with TypeScript
- State Management: 
  - Zustand (lightweight) or Redux Toolkit
  - RTK Query for API calls
- Real-time: 
  - Socket.io-client hoặc native WebSocket
- Charting:
  - Recharts (declarative, simple)
  - D3.js (custom, powerful)
  - Plotly.js (scientific charts)
- 3D Visualization:
  - Three.js + React Three Fiber (trajectory, map view)
- Video:
  - WebRTC (best latency)
  - Video.js (fallback)
- UI Components:
  - Ant Design / Material-UI / shadcn/ui
  - React Grid Layout (customizable dashboard)
```

#### 2.3.4 Backend Services

```
- API Gateway: Kong / Traefik / Nginx
- WebSocket Server: Node.js + Socket.io / Go + Gorilla WebSocket
- REST API: FastAPI (Python) / Fiber (Go) / Express (Node.js)
- Telemetry Service: Go (performance) / Python (easy integration with ML)
- Control Service: Go / Rust (safety-critical, low latency)
- Video Streaming: Go + FFmpeg / Kurento / mediasoup
```

**Recommendation**:
- **Go** cho services cần performance (telemetry, control, video)
- **Python/FastAPI** cho services cần flexibility (API, data processing)
- **Node.js** cho WebSocket server (best ecosystem)

### 2.4 Database Schema Design

#### TimescaleDB Schema

```sql
-- Hypertable for telemetry
CREATE TABLE telemetry (
    time TIMESTAMPTZ NOT NULL,
    vehicle_id VARCHAR(50) NOT NULL,
    speed FLOAT,
    steering_angle FLOAT,
    throttle FLOAT,
    brake FLOAT,
    gear INT,
    latitude FLOAT,
    longitude FLOAT,
    heading FLOAT,
    metadata JSONB  -- flexible for additional sensors
);

SELECT create_hypertable('telemetry', 'time');
CREATE INDEX ON telemetry (vehicle_id, time DESC);

-- Planning data
CREATE TABLE planning_data (
    time TIMESTAMPTZ NOT NULL,
    vehicle_id VARCHAR(50) NOT NULL,
    trajectory JSONB,  -- array of waypoints
    target_speed FLOAT,
    planning_state VARCHAR(50),
    horizon_seconds FLOAT
);

SELECT create_hypertable('planning_data', 'time');

-- Control commands
CREATE TABLE control_commands (
    time TIMESTAMPTZ NOT NULL,
    vehicle_id VARCHAR(50) NOT NULL,
    command_id UUID NOT NULL,
    user_id VARCHAR(50),
    command_type VARCHAR(50),  -- mode_change, param_update, etc.
    payload JSONB,
    ack_time TIMESTAMPTZ,
    status VARCHAR(20)  -- pending, executed, failed, timeout
);

SELECT create_hypertable('control_commands', 'time');
CREATE INDEX ON control_commands (command_id);

-- Events & alerts
CREATE TABLE events (
    time TIMESTAMPTZ NOT NULL,
    vehicle_id VARCHAR(50),
    event_type VARCHAR(50),
    severity VARCHAR(20),  -- info, warning, error, critical
    message TEXT,
    metadata JSONB
);

SELECT create_hypertable('events', 'time');

-- Continuous aggregates for dashboard
CREATE MATERIALIZED VIEW telemetry_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS bucket,
       vehicle_id,
       AVG(speed) as avg_speed,
       MAX(speed) as max_speed,
       AVG(steering_angle) as avg_steering,
       COUNT(*) as sample_count
FROM telemetry
GROUP BY bucket, vehicle_id;

SELECT add_continuous_aggregate_policy('telemetry_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');
```

#### PostgreSQL Schema

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,  -- viewer, operator, admin
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- Sessions (for replay)
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id VARCHAR(50) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    user_id UUID REFERENCES users(id),
    status VARCHAR(20),  -- recording, completed, error
    metadata JSONB,
    video_path VARCHAR(500),
    data_path VARCHAR(500)
);

-- Parameter configs
CREATE TABLE parameter_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

-- Annotations
CREATE TABLE annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    timestamp TIMESTAMPTZ NOT NULL,
    user_id UUID REFERENCES users(id),
    annotation_type VARCHAR(50),
    text TEXT,
    metadata JSONB
);
```

### 2.5 Realtime Optimization Strategies

#### 2.5.1 Latency Reduction

1. **Message Batching với Smart Throttling**
```javascript
// Vehicle side: batch telemetry
const telemetryBuffer = [];
const BATCH_SIZE = 10; // 10 samples
const BATCH_TIMEOUT = 100; // ms

function sendTelemetry(data) {
    telemetryBuffer.push(data);
    if (telemetryBuffer.length >= BATCH_SIZE) {
        flushBatch();
    }
}

setInterval(flushBatch, BATCH_TIMEOUT);
```

2. **WebSocket Optimization**
```javascript
// Server: per-client rate limiting
class ClientConnection {
    constructor(socket) {
        this.socket = socket;
        this.lastSend = 0;
        this.minInterval = 50; // 20 Hz max
    }
    
    sendIfReady(data) {
        const now = Date.now();
        if (now - this.lastSend >= this.minInterval) {
            this.socket.send(data);
            this.lastSend = now;
            return true;
        }
        return false;
    }
}
```

3. **Progressive Data Loading**
- Initial load: last 30s of data
- Background load: historical data
- On-demand: load older data when user scrolls

4. **Binary Protocols**
- Sử dụng **Protobuf** thay vì JSON cho telemetry (giảm 30-50% size)
- **MessagePack** cho WebSocket payloads

#### 2.5.2 Network Optimization

```
┌─────────────────────────────────────────┐
│ Vehicle (Edge)                           │
│  - Local WiFi: 5GHz, 802.11ac           │
│  - Fallback: 4G/LTE                     │
│  - Connection monitoring & auto-switch   │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Edge Router / Access Point               │
│  - QoS: prioritize control commands     │
│  - VLAN isolation for vehicle traffic    │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Backend (Cloud / On-premise)            │
│  - Region: Same as competition venue    │
│  - CDN: Cloudflare for static assets    │
└─────────────────────────────────────────┘
```

### 2.6 Deployment Architecture

#### Option 1: Cloud-Native (AWS/GCP/Azure)

```yaml
# Kubernetes deployment
Services:
  - API Gateway: AWS ALB + Kong (EKS)
  - Microservices: EKS pods (auto-scaling)
  - Message Bus: AWS MSK (Kafka) / NATS on EKS
  - Databases:
      - TimescaleDB: RDS PostgreSQL + Timescale extension
      - Redis: ElastiCache
      - S3: Video & log storage
  - Monitoring: CloudWatch + Prometheus + Grafana
  
Pros:
  - Scalability
  - Managed services
  - Global reach
  
Cons:
  - Cost
  - Latency (if far from venue)
  - Vendor lock-in
```

#### Option 2: On-Premise (Recommended for Competition)

```yaml
# Docker Compose / Kubernetes (K3s)
Hardware:
  - Server: Dell PowerEdge / HP ProLiant
  - CPU: 16+ cores
  - RAM: 64GB+
  - Storage: 1TB NVMe SSD + 4TB HDD
  - Network: 10GbE
  
Services:
  - All services containerized
  - NATS for message bus
  - TimescaleDB + Redis on bare metal (performance)
  - MinIO for object storage
  - Monitoring: Prometheus + Grafana
  
Pros:
  - Low latency (same network as vehicle)
  - No cloud cost
  - Full control
  
Cons:
  - Setup complexity
  - No auto-scaling
  - Single point of failure (mitigate with HA setup)
```

**Recommendation**: **Hybrid approach**
- On-premise server tại venue (primary)
- Cloud backup (secondary, for remote access)
- Sync data to cloud post-competition for analysis

### 2.7 Monitoring & Observability

#### 2.7.1 Metrics to Track

**Application Metrics** (Prometheus):
```yaml
# Telemetry Service
- telemetry_messages_received_total (counter)
- telemetry_processing_latency_seconds (histogram)
- telemetry_validation_errors_total (counter)

# Control Service
- control_commands_sent_total (counter)
- control_command_latency_seconds (histogram)
- control_commands_failed_total (counter)
- control_commands_duplicate_total (counter)

# WebSocket Server
- websocket_connections_active (gauge)
- websocket_messages_sent_total (counter)
- websocket_broadcast_latency_seconds (histogram)

# Video Streaming
- video_bitrate_kbps (gauge)
- video_frames_dropped_total (counter)
- video_encoding_latency_seconds (histogram)
```

**Infrastructure Metrics**:
- CPU, memory, disk, network per service
- Message bus lag (Kafka consumer lag / NATS pending)
- Database connection pool usage

#### 2.7.2 Logging Strategy

```yaml
Structured Logging (JSON):
  - timestamp
  - level (debug, info, warn, error)
  - service_name
  - trace_id (distributed tracing)
  - vehicle_id (contextual)
  - message
  - metadata (arbitrary fields)

Levels:
  - DEBUG: Development only
  - INFO: Important state changes (mode switch, connection, etc.)
  - WARN: Recoverable errors (retry, timeout)
  - ERROR: Unrecoverable errors
  
Aggregation:
  - Loki (lightweight) or ELK stack
  - Retention: 30 days
```

#### 2.7.3 Distributed Tracing

```yaml
Tool: Jaeger / Tempo

Trace Flow Example:
  1. User clicks "Emergency Stop" on web
     - Span: web_ui.emergency_stop_click
  2. WebSocket sends command
     - Span: websocket.send_command
  3. Control Service receives
     - Span: control_svc.receive_command
  4. Validation + Kafka publish
     - Span: control_svc.validate
     - Span: kafka.publish
  5. Vehicle receives & executes
     - Span: vehicle.execute_command
  6. ACK back
     - Span: vehicle.send_ack
  
Total trace time = End-to-end latency
Can identify bottleneck at each span
```

#### 2.7.4 Alerting Rules

```yaml
# Prometheus Alertmanager rules
groups:
  - name: vehicle_alerts
    rules:
      - alert: VehicleDisconnected
        expr: up{job="vehicle_agent"} == 0
        for: 10s
        annotations:
          summary: "Vehicle {{ $labels.vehicle_id }} disconnected"
          
      - alert: HighTelemetryLatency
        expr: histogram_quantile(0.95, telemetry_processing_latency_seconds) > 0.2
        for: 1m
        annotations:
          summary: "Telemetry latency P95 > 200ms"
          
      - alert: ControlCommandTimeout
        expr: rate(control_commands_failed_total{reason="timeout"}[5m]) > 0.1
        annotations:
          summary: "Control commands timing out"
          
      - alert: LowBattery
        expr: vehicle_battery_percent < 20
        annotations:
          summary: "Vehicle {{ $labels.vehicle_id }} battery low"
```

### 2.8 Security Considerations

```yaml
Authentication:
  - JWT tokens (access + refresh)
  - Access token expiry: 15 minutes
  - Refresh token expiry: 7 days
  - Secure storage: httpOnly cookies or secure localStorage

Authorization:
  - RBAC (Role-Based Access Control)
  - Permissions checked at API gateway + service level
  - Audit log for all control commands

Communication Security:
  - TLS 1.3 for all HTTPS/WSS
  - Certificate pinning for vehicle ↔ backend
  - mTLS for service-to-service (optional, trong cluster)

Input Validation:
  - All user inputs validated (frontend + backend)
  - Command payload schema validation
  - Rate limiting per user (100 req/min)

Secrets Management:
  - Environment variables (dev)
  - HashiCorp Vault / AWS Secrets Manager (prod)
  - Rotate credentials regularly
```
---
