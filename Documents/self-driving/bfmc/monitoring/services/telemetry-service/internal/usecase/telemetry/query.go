package telemetry

import (
	"context"

	"telemetry-service/internal/domain/telemetry"
)

type QueryUseCase struct {
	repo telemetry.Repository
}

func NewQueryUseCase(repo telemetry.Repository) *QueryUseCase {
	return &QueryUseCase{repo: repo}
}

func (uc *QueryUseCase) GetLatest(ctx context.Context, vehicleID string) (*telemetry.Telemetry, error) {
	return uc.repo.GetLatest(ctx, vehicleID)
}
