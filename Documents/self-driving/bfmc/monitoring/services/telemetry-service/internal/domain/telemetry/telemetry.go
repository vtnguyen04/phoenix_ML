package telemetry

import (
    "time"
    "errors"
)

var (
    ErrInvalidVehicleID = errors.New("invalid vehicle ID")
    ErrInvalidSpeed     = errors.New("invalid speed")
)

// Telemetry represents vehicle telemetry data (Domain Entity)
type Telemetry struct {
    Timestamp      time.Time
    VehicleID      string
    Speed          float64
    SteeringAngle  float64
    Throttle       float64
    Brake          float64
    Gear           int
    Location       Location
    Metadata       map[string]interface{}
}

type Location struct {
    Latitude  float64
    Longitude float64
    Heading   float64
}

// Validate performs domain validation
func (t *Telemetry) Validate() error {
    if t.VehicleID == "" {
        return ErrInvalidVehicleID
    }
    if t.Speed < 0 {
        return ErrInvalidSpeed
    }
    // ... more validations
    return nil
}

// IsEmergency checks if telemetry indicates emergency
func (t *Telemetry) IsEmergency() bool {
    return t.Speed > 100 || t.Brake > 0.9
}
