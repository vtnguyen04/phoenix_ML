package timescaledb

import (
    "context"
    "database/sql"
    "telemetry-service/internal/domain/telemetry"
	"time"
)

type TelemetryRepository struct {
    db *sql.DB
}

func NewTelemetryRepository(db *sql.DB) *TelemetryRepository {
    return &TelemetryRepository{db: db}
}

func (r *TelemetryRepository) Store(ctx context.Context, t *telemetry.Telemetry) error {
    query := `
        INSERT INTO telemetry (
            time, vehicle_id, speed, steering_angle, throttle, brake, gear,
            latitude, longitude, heading, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    `
    
    _, err := r.db.ExecContext(ctx, query,
        t.Timestamp,
        t.VehicleID,
        t.Speed,
        t.SteeringAngle,
        t.Throttle,
        t.Brake,
        t.Gear,
        t.Location.Latitude,
        t.Location.Longitude,
        t.Location.Heading,
        t.Metadata,
    )
    
    return err
}

func (r *TelemetryRepository) GetLatest(ctx context.Context, vehicleID string) (*telemetry.Telemetry, error) {
    query := `
        SELECT time, vehicle_id, speed, steering_angle, throttle, brake, gear,
               latitude, longitude, heading, metadata
        FROM telemetry
        WHERE vehicle_id = $1
        ORDER BY time DESC
        LIMIT 1
    `
    
    var t telemetry.Telemetry
    // Note: The Location field is a struct, so you'll need to scan the individual fields
    // and assemble the struct. The Metadata field is JSONB, which can be scanned into a []byte or string
    // and then unmarshalled. This is a simplified example.
    err := r.db.QueryRowContext(ctx, query, vehicleID).Scan(
        &t.Timestamp,
        &t.VehicleID,
        &t.Speed,
        &t.SteeringAngle,
        &t.Throttle,
        &t.Brake,
        &t.Gear,
        &t.Location.Latitude,
        &t.Location.Longitude,
        &t.Location.Heading,
        &t.Metadata,
    )
    
    if err != nil {
        return nil, err
    }
    
    return &t, nil
}

// StoreBatch, GetByTimeRange, and GetAggregated would also be implemented here.
func (r *TelemetryRepository) StoreBatch(ctx context.Context, telemetries []*telemetry.Telemetry) error {
    // Implementation for batch storing telemetry data
    return nil
}

func (r *TelemetryRepository) GetByTimeRange(ctx context.Context, vehicleID string, start, end time.Time) ([]*telemetry.Telemetry, error) {
    // Implementation for getting telemetry data by time range
    return nil, nil
}

func (r *TelemetryRepository) GetAggregated(ctx context.Context, vehicleID string, interval time.Duration) ([]*telemetry.AggregatedTelemetry, error) {
    // Implementation for getting aggregated telemetry data
    return nil, nil
}
