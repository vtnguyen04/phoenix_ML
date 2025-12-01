import React, { useState, useEffect } from 'react';
import Header from '../shared/components/layout/Header';
import Sidebar from '../shared/components/layout/Sidebar';
import Footer from '../shared/components/layout/Footer';
import { fetchLatestTelemetry } from '../core/api/telemetryClient';

interface TelemetryData {
  Timestamp: string;
  VehicleID: string;
  Speed: number;
  SteeringAngle: number;
  Throttle: number;
  Brake: number;
  Gear: number;
  Location: {
    Latitude: number;
    Longitude: number;
    Heading: number;
  };
  Metadata: any;
}

function App() {
  const [latestTelemetry, setLatestTelemetry] = useState<TelemetryData | null>(null);
  const vehicleID = "test-vehicle"; // Hardcoded for now

  useEffect(() => {
    const getTelemetry = async () => {
      const data = await fetchLatestTelemetry(vehicleID);
      if (data) {
        setLatestTelemetry(data);
      }
    };

    // Fetch immediately and then every 1 second
    getTelemetry();
    const intervalId = setInterval(getTelemetry, 1000);

    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [vehicleID]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Header />
      <div style={{ display: 'flex', flexGrow: 1 }}>
        <Sidebar />
        <main style={{ flexGrow: 1, padding: '1rem' }}>
          <h1>Vehicle Monitoring Dashboard</h1>
          <h2>Latest Telemetry ({vehicleID})</h2>
          {latestTelemetry ? (
            <div>
              <p>Speed: {latestTelemetry.Speed} km/h</p>
              <p>Steering Angle: {latestTelemetry.SteeringAngle}°</p>
              <p>Throttle: {latestTelemetry.Throttle}</p>
              <p>Brake: {latestTelemetry.Brake}</p>
              <p>Timestamp: {new Date(latestTelemetry.Timestamp).toLocaleString()}</p>
            </div>
          ) : (
            <p>Loading telemetry data...</p>
          )}
        </main>
      </div>
      <Footer />
    </div>
  );
}

export default App;
