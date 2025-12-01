package telemetry

import (
	"context"

	"telemetry-service/internal/domain/telemetry"
)

type MessagePublisher interface {
	PublishEmergency(ctx context.Context, data *telemetry.Telemetry) error
}

type MetricsClient interface {
	IncrementCounter(name string)
	RecordHistogram(name string, value float64)
}
