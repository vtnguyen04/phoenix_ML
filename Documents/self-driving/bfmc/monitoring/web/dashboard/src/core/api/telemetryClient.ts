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

export const fetchLatestTelemetry = async (vehicleID: string): Promise<TelemetryData | null> => {
  try {
    const response = await fetch(`http://localhost:8080/telemetry/latest?vehicle_id=${vehicleID}`);
    if (!response.ok) {
      console.error('Failed to fetch latest telemetry:', response.statusText);
      return null;
    }
    const data: TelemetryData = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching latest telemetry:', error);
    return null;
  }
};
