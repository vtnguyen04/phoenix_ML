package telemetry

import (
    "context"
    "time"

    "telemetry-service/internal/domain/telemetry"
    	"telemetry-service/internal/infrastructure/logger")

type IngestUseCase struct {
    repo          telemetry.Repository
    cache         telemetry.CacheRepository
    publisher     MessagePublisher
    logger        logger.Logger
    metricsClient MetricsClient
}

func NewIngestUseCase(
    repo telemetry.Repository,
    cache telemetry.CacheRepository,
    publisher MessagePublisher,
    logger logger.Logger,
    metrics MetricsClient,
) *IngestUseCase {
    return &IngestUseCase{
        repo:          repo,
        cache:         cache,
        publisher:     publisher,
        logger:        logger,
        metricsClient: metrics,
    }
}

func (uc *IngestUseCase) Execute(ctx context.Context, data *telemetry.Telemetry) error {
    // 1. Validate domain rules
    if err := data.Validate(); err != nil {
        uc.metricsClient.IncrementCounter("telemetry.validation.errors")
        return err
    }
    
    // 2. Store in database
    if err := uc.repo.Store(ctx, data); err != nil {
        uc.logger.Error("failed to store telemetry", "error", err)
        return err
    }
    
    // 3. Update cache (latest value)
    cacheKey := "telemetry:latest:" + data.VehicleID
    if err := uc.cache.Set(ctx, cacheKey, data, 10*time.Minute); err != nil {
        uc.logger.Warn("failed to update cache", "error", err)
        // Non-critical, continue
    }
    
    // 4. Publish event for downstream consumers
    if data.IsEmergency() {
        if err := uc.publisher.PublishEmergency(ctx, data); err != nil {
            uc.logger.Error("failed to publish emergency event", "error", err)
        }
    }
    
    // 5. Record metrics
    uc.metricsClient.RecordHistogram("telemetry.ingest.latency", time.Since(data.Timestamp).Seconds())
    uc.metricsClient.IncrementCounter("telemetry.ingested.total")
    
    return nil
}
