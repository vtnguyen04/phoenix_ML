## III. IMPLEMENTATION ROADMAP (Simplified for High-Performance MVP)

### Phase 1: High-Performance MVP (1-2 weeks)
- [x] Clean up existing project structure (removed NATS, TimescaleDB, Redis, etc.)
- [x] Re-architected system for UDP/WebSocket direct communication
- [ ] Implement Go Backend Service:
  - [ ] Serve static frontend files (HTTP)
  - [ ] Implement WebSocket server
  - [ ] Implement UDP listener (telemetry input)
  - [ ] Forward UDP telemetry to WebSocket clients
- [ ] Implement Vehicle Agent Simulator:
  - [ ] Generate basic telemetry data
  - [ ] Send telemetry via UDP to Go Backend
- [ ] Implement Frontend (Vite + React):
  - [ ] Connect to Go Backend via WebSocket
  - [ ] Display real-time telemetry data (speed, steering, etc.)
  - [ ] Basic dashboard layout

### Phase 2: Core Features (TBD)
- [ ] Control command flow (via WebSocket/UDP)
- [ ] Basic video streaming (e.g., MJPEG over WebSocket)
- [ ] Parameter tuning UI
- [ ] Simple data logging for short-term replay (e.g., local file or in-memory)

### Phase 3: Further Enhancements (TBD)
- [ ] Advanced visualization
- [ ] Alerting and notifications

---

## IV. RISK MITIGATION

(Adapted for simplified architecture)

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| UDP Packet Loss | Medium | Low | - Acknowledge acceptable for real-time monitoring<br>- Frontend can display "last known good" data<br>- Implement simple retransmission for critical commands if needed in future phases |
| Single Point of Failure (Go Backend) | High | Low | - Auto-restart mechanisms for Go service<br>- Detailed logging for quick diagnostics |
| Performance bottlenecks (Go/Frontend) | Low | Low | - Profile Go backend and frontend code<br>- Optimize data serialization/deserialization |
| Environment setup issues | High | Medium | - Focus on minimal external dependencies<br>- Provide clear, concise run instructions |

---