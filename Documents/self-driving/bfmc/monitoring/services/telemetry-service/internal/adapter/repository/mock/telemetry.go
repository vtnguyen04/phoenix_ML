package mock

import (
	"context"
	"telemetry-service/internal/domain/telemetry"
	"time"
)

type TelemetryRepository struct{}

func NewTelemetryRepository() *TelemetryRepository {
	return &TelemetryRepository{}
}

func (r *TelemetryRepository) Store(ctx context.Context, t *telemetry.Telemetry) error {
	return nil
}

func (r *TelemetryRepository) GetLatest(ctx context.Context, vehicleID string) (*telemetry.Telemetry, error) {
	return &telemetry.Telemetry{
		VehicleID: vehicleID,
		Speed:     60,
		Timestamp: time.Now(),
	}, nil
}

func (r *TelemetryRepository) StoreBatch(ctx context.Context, telemetries []*telemetry.Telemetry) error {
	return nil
}

func (r *TelemetryRepository) GetByTimeRange(ctx context.Context, vehicleID string, start, end time.Time) ([]*telemetry.Telemetry, error) {
	return nil, nil
}

func (r *TelemetryRepository) GetAggregated(ctx context.Context, vehicleID string, interval time.Duration) ([]*telemetry.AggregatedTelemetry, error) {
	return nil, nil
}
