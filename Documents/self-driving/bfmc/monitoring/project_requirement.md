# Hệ thống Realtime Monitoring cho Xe Tự Lái 1:10 - Bosch Future Mobility Challenge

## I. PROJECT REQUIREMENTS

### 1.1 Mô tả Hệ thống

Hệ thống web-based realtime monitoring và control platform cho xe tự lái tỉ lệ 1:10, cho phép giám sát toàn diện hoạt động của xe, phân tích hiệu suất các module AI/planning, và điều khiển/tinh chỉnh tham số trong thời gian thực. Hệ thống được thiết kế theo kiến trúc microservices với khả năng mở rộng cao, đáp ứng yêu cầu độ trễ thấp (<100ms cho control commands) và tính sẵn sàng cao.

### 1.2 Use Cases Chính

**UC1: Realtime Monitoring (Viewer/Operator/Admin)**
- Xem trạng thái xe realtime: tốc độ, góc lái, throttle, brake
- Theo dõi video stream với object detection overlay
- Giám sát health metrics: FPS, latency, nhiệt độ, pin, kết nối
- Xem planner trajectory và so sánh với actual path

**UC2: Performance Analysis (Operator/Admin)**
- Phân tích time-series data của các tín hiệu điều khiển
- So sánh planned vs actual values (tốc độ, steering angle)
- Xác định bottleneck và lỗi trong pipeline
- Đánh giá hiệu suất perception module (detection accuracy, FPS)

**UC3: Remote Control (Operator/Admin)**
- Chuyển đổi chế độ: Manual/Autonomous/Emergency Stop
- Gửi lệnh điều khiển cơ bản: start/stop, speed limit
- Emergency brake từ xa
- Xác nhận và log mọi thao tác điều khiển

**UC4: Parameter Tuning (Admin)**
- Điều chỉnh PID gains cho steering/throttle controller
- Thay đổi planner parameters (lookahead distance, planning horizon)
- Set speed limits, safety thresholds
- A/B testing các bộ tham số khác nhau

**UC5: Data Logging & Replay (Operator/Admin)**
- Ghi lại toàn bộ session (telemetry, video metadata, events)
- Replay session với timeline control (play/pause/seek)
- Export data để phân tích offline
- Annotate events/issues trong timeline

**UC6: Multi-Vehicle Fleet Management (Admin)** *(Mở rộng tương lai)*
- Giám sát nhiều xe đồng thời
- So sánh performance cross-vehicle
- Deploy configuration updates đồng loạt

### 1.3 User Roles & Permissions

| Role | Permissions |
|------|-------------|
| **Viewer** | - Xem dashboard realtime<br>- Xem video stream<br>- Xem historical data<br>- Export reports |
| **Operator** | - Tất cả quyền Viewer<br>- Remote control (start/stop, mode switch)<br>- Emergency stop<br>- Replay sessions |
| **Admin** | - Tất cả quyền Operator<br>- Parameter tuning<br>- User management<br>- System configuration<br>- Access raw logs |

### 1.4 Functional Requirements

#### FR1: Dashboard Giám Sát Tổng Quan
- **FR1.1**: Hiển thị time-series charts cho speed, steering angle, throttle, brake (update rate: 10-20 Hz)
- **FR1.2**: Status cards hiển thị:
  - Connection status (connected/disconnected/reconnecting)
  - System health (CPU, memory, temperature)
  - Battery level & voltage
  - Current mode (manual/autonomous)
  - Active warnings/errors count
- **FR1.3**: Performance metrics:
  - End-to-end latency (perception → planning → control)
  - Individual module FPS (camera, detection, planning, control)
  - Network latency (vehicle ↔ backend ↔ web)
- **FR1.4**: Customizable layout (drag-and-drop widgets, save presets)

#### FR2: Camera & Object Detection Visualization
- **FR2.1**: Live video stream (target: 20-30 FPS, max latency: 200ms)
- **FR2.2**: Overlay rendering:
  - Bounding boxes với label + confidence score
  - Lane detection lines
  - Planned trajectory projection
  - Velocity vectors của detected objects
- **FR2.3**: Stream controls: pause, snapshot, record clip
- **FR2.4**: Multi-camera support (nếu xe có nhiều camera)
- **FR2.5**: Stream quality adaptation (auto/manual bitrate control)

#### FR3: Planning & Control Monitoring
- **FR3.1**: Visualize planned trajectory:
  - Path visualization trên bird's eye view
  - Waypoints với timestamp dự kiến
  - Speed profile along trajectory
- **FR3.2**: Comparison charts:
  - Target speed vs Actual speed
  - Commanded steering vs Actual steering
  - Planned position vs Actual position (nếu có localization)
- **FR3.3**: Planning state machine visualization
- **FR3.4**: Control loop diagnostics:
  - PID terms (P, I, D components)
  - Control error over time
  - Saturation indicators

#### FR4: Realtime Control & Tuning
- **FR4.1**: Mode control với confirmation dialog:
  - Switch Manual/Autonomous
  - Emergency Stop (no confirmation, instant)
  - Start/Stop engine
- **FR4.2**: Parameter tuning interface:
  - Live parameter editor với validation
  - Preset management (save/load/share)
  - Rollback mechanism
  - Change history log
- **FR4.3**: Safety features:
  - Command rate limiting
  - Duplicate command detection
  - Timeout & watchdog
  - Geo-fencing (nếu áp dụng)
- **FR4.4**: Manual override controls (joystick/keyboard cho emergency)

#### FR5: Logging, Replay & Analysis
- **FR5.1**: Auto-logging với configurable triggers:
  - Continuous recording
  - Event-triggered (errors, emergency stops)
  - Manual start/stop
- **FR5.2**: Data logging scope:
  - Full telemetry (all sensor data, control commands)
  - Video (configurable: full video, keyframes only, or metadata)
  - Events & annotations
  - System logs
- **FR5.3**: Replay functionality:
  - Timeline với playback controls
  - Variable speed playback (0.25x - 4x)
  - Frame-by-frame stepping
  - Synchronized multi-stream replay
- **FR5.4**: Analysis tools:
  - Chart overlay trên timeline
  - Event markers & annotations
  - Data export (CSV, JSON, ROS bag format)
  - Statistical summary reports

#### FR6: Alerting & Notifications
- **FR6.1**: Configurable alerts:
  - System health thresholds (temp, battery, CPU)
  - Performance degradation (FPS drop, high latency)
  - Safety violations (speed limit, geo-fence)
  - Module failures
- **FR6.2**: Notification channels:
  - In-dashboard notifications
  - Email alerts
  - Webhook integration (Slack, Discord...)
- **FR6.3**: Alert history & acknowledgment

### 1.5 Non-Functional Requirements

#### NFR1: Performance & Latency
- **NFR1.1**: Control command latency: <50ms (web → backend → vehicle)
- **NFR1.2**: Telemetry update latency: <100ms (vehicle → backend → web)
- **NFR1.3**: Video streaming latency: <200ms (acceptable), <500ms (degraded)
- **NFR1.4**: Dashboard update rate: 10-20 Hz cho critical metrics, 1-5 Hz cho non-critical
- **NFR1.5**: Page load time: <2s (initial), <500ms (subsequent navigation)

#### NFR2: Reliability & Availability
- **NFR2.1**: System uptime: 99.5% (cho test/competition environment)
- **NFR2.2**: Graceful degradation khi mất kết nối:
  - Auto-reconnect với exponential backoff
  - Local buffering của commands chưa gửi được
  - Fallback display mode (last known state)
- **NFR2.3**: Data consistency:
  - At-least-once delivery cho critical commands
  - Idempotent command handling
  - Sequence numbering & duplicate detection
- **NFR2.4**: Fault tolerance:
  - Service health checks
  - Automatic service restart
  - Circuit breaker pattern

#### NFR3: Security
- **NFR3.1**: Authentication: JWT-based với refresh token
- **NFR3.2**: Authorization: Role-based access control (RBAC)
- **NFR3.3**: Communication security:
  - TLS/SSL cho web traffic
  - Encrypted WebSocket connections
  - Certificate pinning cho vehicle ↔ backend
- **NFR3.4**: Audit logging cho tất cả control commands
- **NFR3.5**: Rate limiting per user/role

#### NFR4: Scalability
- **NFR4.1**: Hỗ trợ multiple concurrent viewers (10-50 users)
- **NFR4.2**: Horizontal scaling cho stateless services
- **NFR4.3**: Storage scaling:
  - Time-series data: 100K points/second
  - Video storage: configurable retention (1-30 days)
- **NFR4.4**: Extensibility:
  - Plugin architecture cho custom widgets
  - API-first design cho integration
  - Schema versioning

#### NFR5: Observability
- **NFR5.1**: Structured logging (JSON format)
- **NFR5.2**: Metrics collection:
  - Business metrics (active users, command count)
  - Technical metrics (latency, throughput, error rate)
  - Infrastructure metrics (CPU, memory, network)
- **NFR5.3**: Distributed tracing cho request flows
- **NFR5.4**: Health check endpoints cho tất cả services

#### NFR6: Usability
- **NFR6.1**: Responsive design (desktop, tablet)
- **NFR6.2**: Dark/light theme support
- **NFR6.3**: Keyboard shortcuts cho common actions
- **NFR6.4**: Real-time help tooltips
- **NFR6.5**: Multi-language support (EN, VI) *(optional)*