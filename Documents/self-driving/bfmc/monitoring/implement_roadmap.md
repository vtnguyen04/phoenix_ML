
## III. IMPLEMENTATION ROADMAP

### Phase 1: MVP (2-3 weeks)
- [ ] Setup infrastructure (Docker Compose, NATS, TimescaleDB, Redis)
- [ ] Vehicle agent với basic telemetry publishing
- [ ] Backend: Telemetry service + WebSocket server
- [ ] Frontend: Basic dashboard với live charts
- [ ] Basic video streaming (WebSocket + MJPEG)

### Phase 2: Core Features (2-3 weeks)
- [ ] Control command flow với safety checks
- [ ] Planning data visualization
- [ ] Parameter tuning UI
- [ ] Authentication & authorization
- [ ] Logging & replay (basic)

### Phase 3: Polish & Optimization (2 weeks)
- [ ] Latency optimization
- [ ] Advanced video streaming (WebRTC)
- [ ] Alert system
- [ ] Monitoring & observability stack
- [ ] Load testing & performance tuning

### Phase 4: Competition Ready (1 week)
- [ ] On-premise deployment setup
- [ ] Failover & redundancy
- [ ] Documentation & runbooks
- [ ] Dry run & stress testing

---

## IV. RISK MITIGATION

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Network instability at venue | High | Medium | - Multi-connection fallback (WiFi + 4G)<br>- Local buffering<br>- Graceful degradation |
| High latency breaking control | High | Medium | - Aggressive timeout<br>- Local autonomy mode<br>- Emergency stop always available |
| Video stream bandwidth saturation | Medium | High | - Adaptive bitrate<br>- Priority QoS for control<br>- Disable video nếu cần |
| Database overload | Medium | Low | - Connection pooling<br>- Rate limiting writes<br>- Aggregation before storage |
| Service crashes | High | Low | - Health checks<br>- Auto-restart<br>- Circuit breaker pattern |

---