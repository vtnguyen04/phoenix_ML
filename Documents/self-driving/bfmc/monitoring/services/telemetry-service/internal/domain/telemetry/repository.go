package telemetry

import (
    "context"
    "time"
)

type AggregatedTelemetry struct {
    // Define your aggregated telemetry fields
}

// Repository defines data access interface (Port)
type Repository interface {
    // Write operations
    Store(ctx context.Context, telemetry *Telemetry) error
    StoreBatch(ctx context.Context, telemetries []*Telemetry) error
    
    // Read operations
    GetLatest(ctx context.Context, vehicleID string) (*Telemetry, error)
    GetByTimeRange(ctx context.Context, vehicleID string, start, end time.Time) ([]*Telemetry, error)
    GetAggregated(ctx context.Context, vehicleID string, interval time.Duration) ([]*AggregatedTelemetry, error)
}

// CacheRepository for caching layer
type CacheRepository interface {
    Get(ctx context.Context, key string) (*Telemetry, error)
    Set(ctx context.Context, key string, telemetry *Telemetry, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
}
