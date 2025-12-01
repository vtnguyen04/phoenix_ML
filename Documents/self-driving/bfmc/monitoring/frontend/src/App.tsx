import React, { useState, useEffect } from 'react';

interface TelemetryData {
  Timestamp: string;
  VehicleID: string;
  Speed: number;
  SteeringAngle: number;
  Throttle: number;
  Brake: number;
  Gear: number;
  Latitude: number;
  Longitude: number;
  Heading: number;
  Metadata: any;
}

function App() {
  const [latestTelemetry, setLatestTelemetry] = useState<TelemetryData | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8080/ws");

    ws.onmessage = (event) => {
      const data: TelemetryData = JSON.parse(event.data);
      setLatestTelemetry(data);
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <header style={{ padding: '1rem', background: '#333', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Vehicle Monitoring Platform</h2>
        <div>Status: {latestTelemetry ? 'Connected' : 'Disconnected'}</div>
      </header>
      <div style={{ display: 'flex', flexGrow: 1 }}>
        <aside style={{ width: '200px', background: '#f0f0f0', padding: '1rem', height: 'calc(100vh - 64px)' }}>
          <nav>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '1rem' }}><a href="#" style={{ textDecoration: 'none', color: '#333' }}>Dashboard</a></li>
            </ul>
          </nav>
        </aside>
        <main style={{ flexGrow: 1, padding: '1rem' }}>
          <h1>Dashboard</h1>
          {latestTelemetry ? (
            <div>
              <p>Speed: {latestTelemetry.Speed.toFixed(2)} km/h</p>
              <p>Steering Angle: {latestTelemetry.SteeringAngle.toFixed(2)}°</p>
              <p>Throttle: {latestTelemetry.Throttle.toFixed(2)}</p>
              <p>Brake: {latestTelemetry.Brake.toFixed(2)}</p>
              <p>Timestamp: {new Date(latestTelemetry.Timestamp).toLocaleString()}</p>
            </div>
          ) : (
            <p>Waiting for telemetry data...</p>
          )}
        </main>
      </div>
      <footer style={{ padding: '1rem', background: '#333', color: 'white', textAlign: 'center' }}>
        <p>&copy; 2025 Vehicle Monitoring Platform</p>
      </footer>
    </div>
  );
}

export default App;
