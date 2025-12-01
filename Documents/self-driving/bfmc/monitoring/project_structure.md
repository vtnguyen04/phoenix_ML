# Project Structure - Realtime Vehicle Monitoring System

## I. MONOREPO STRUCTURE (Recommended)

```
vehicle-monitoring-platform/
├── README.md
├── .gitignore
├── .editorconfig
├── docker-compose.yml
├── docker-compose.prod.yml
├── Makefile
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── pr-checks.yml
│
├── docs/
│   ├── architecture/
│   │   ├── system-design.md
│   │   ├── data-flow.md
│   │   └── deployment.md
│   ├── api/
│   │   ├── rest-api.md
│   │   ├── websocket-protocol.md
│   │   └── message-schemas.md
│   ├── development/
│   │   ├── setup-guide.md
│   │   ├── coding-standards.md
│   │   └── testing-guide.md
│   └── operations/
│       ├── deployment-guide.md
│       ├── monitoring-guide.md
│       └── troubleshooting.md
│
├── infrastructure/
│   ├── docker/
│   │   ├── base/
│   │   │   ├── node.Dockerfile
│   │   │   ├── golang.Dockerfile
│   │   │   └── python.Dockerfile
│   │   └── services/
│   │       ├── telemetry-service.Dockerfile
│   │       ├── control-service.Dockerfile
│   │       └── websocket-server.Dockerfile
│   ├── kubernetes/
│   │   ├── base/
│   │   │   ├── namespace.yaml
│   │   │   ├── configmap.yaml
│   │   │   └── secrets.yaml
│   │   ├── services/
│   │   │   ├── telemetry-service/
│   │   │   ├── control-service/
│   │   │   └── websocket-server/
│   │   └── deployments/
│   │       ├── development/
│   │       ├── staging/
│   │       └── production/
│   ├── terraform/
│   │   ├── modules/
│   │   ├── environments/
│   │   └── main.tf
│   └── ansible/
│       ├── playbooks/
│       └── roles/
│
├── shared/
│   ├── proto/                    # Protocol Buffers definitions
│   │   ├── telemetry.proto
│   │   ├── control.proto
│   │   ├── planning.proto
│   │   └── common.proto
│   ├── types/                    # Shared TypeScript types
│   │   ├── telemetry.ts
│   │   ├── control.ts
│   │   └── index.ts
│   └── configs/                  # Shared configurations
│       ├── logging.yaml
│       └── monitoring.yaml
│
├── services/
│   ├── telemetry-service/        # Go service
│   ├── control-service/          # Go service
│   ├── video-service/            # Go service
│   ├── websocket-server/         # Node.js service
│   ├── api-gateway/              # Kong/custom
│   ├── auth-service/             # Go service
│   └── data-archiver/            # Python service
│
├── web/
│   └── dashboard/                # React frontend
│
├── vehicle/
│   └── agent/                    # Python/C++ vehicle agent
│
├── scripts/
│   ├── setup/
│   │   ├── init-dev.sh
│   │   └── init-db.sh
│   ├── build/
│   │   ├── build-all.sh
│   │   └── build-proto.sh
│   └── deploy/
│       ├── deploy-dev.sh
│       └── deploy-prod.sh
│
└── tests/
    ├── integration/
    ├── e2e/
    └── performance/
```

---

## II. SERVICE-LEVEL STRUCTURE (Go Services)

### A. Telemetry Service (Clean Architecture)

```
services/telemetry-service/
├── cmd/
│   └── server/
│       └── main.go                    # Entry point
│
├── internal/                          # Private application code
│   ├── domain/                        # Enterprise business rules
│   │   ├── telemetry/
│   │   │   ├── telemetry.go          # Domain entities
│   │   │   ├── repository.go         # Repository interface
│   │   │   └── service.go            # Domain service interface
│   │   └── errors/
│   │       └── errors.go             # Domain errors
│   │
│   ├── usecase/                       # Application business rules
│   │   ├── telemetry/
│   │   │   ├── ingest.go             # Ingest telemetry use case
│   │   │   ├── process.go            # Process telemetry use case
│   │   │   ├── query.go              # Query telemetry use case
│   │   │   └── service.go            # Use case implementation
│   │   └── interfaces.go             # Use case interfaces
│   │
│   ├── adapter/                       # Interface adapters
│   │   ├── repository/
│   │   │   ├── timescaledb/
│   │   │   │   ├── telemetry.go      # TimescaleDB implementation
│   │   │   │   ├── connection.go     # DB connection pool
│   │   │   │   └── migrations/
│   │   │   └── redis/
│   │   │       └── cache.go          # Redis cache implementation
│   │   │
│   │   ├── messaging/
│   │   │   ├── nats/
│   │   │   │   ├── consumer.go       # NATS consumer
│   │   │   │   ├── publisher.go      # NATS publisher
│   │   │   │   └── connection.go
│   │   │   └── kafka/                # Alternative implementation
│   │   │       └── consumer.go
│   │   │
│   │   └── http/
│   │       ├── handler/
│   │       │   ├── telemetry.go      # HTTP handlers
│   │       │   ├── health.go
│   │       │   └── middleware/
│   │       │       ├── auth.go
│   │       │       ├── logging.go
│   │       │       └── metrics.go
│   │       └── router.go
│   │
│   ├── infrastructure/                # Frameworks & drivers
│   │   ├── config/
│   │   │   ├── config.go             # Configuration loading
│   │   │   └── validation.go
│   │   ├── logger/
│   │   │   └── logger.go             # Structured logging
│   │   ├── metrics/
│   │   │   └── prometheus.go         # Prometheus metrics
│   │   ├── tracing/
│   │   │   └── jaeger.go             # Distributed tracing
│   │   └── graceful/
│   │       └── shutdown.go           # Graceful shutdown handler
│   │
│   └── di/                            # Dependency injection
│       └── container.go               # Wire up dependencies
│
├── pkg/                               # Public libraries (reusable)
│   ├── validator/
│   │   └── telemetry.go              # Telemetry validation logic
│   └── converter/
│       └── units.go                  # Unit conversion utilities
│
├── api/
│   ├── grpc/
│   │   └── telemetry.proto
│   └── http/
│       └── openapi.yaml
│
├── configs/
│   ├── config.yaml
│   ├── config.dev.yaml
│   └── config.prod.yaml
│
├── migrations/
│   ├── 001_create_telemetry_table.up.sql
│   ├── 001_create_telemetry_table.down.sql
│   └── ...
│
├── test/
│   ├── unit/
│   │   ├── domain/
│   │   ├── usecase/
│   │   └── adapter/
│   ├── integration/
│   │   ├── repository_test.go
│   │   └── messaging_test.go
│   └── fixtures/
│       └── telemetry.json
│
├── scripts/
│   ├── build.sh
│   └── migrate.sh
│
├── .env.example
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

**Key Files Examples:**

#### `internal/domain/telemetry/telemetry.go`
```go
package telemetry

import (
    "time"
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
```

#### `internal/domain/telemetry/repository.go`
```go
package telemetry

import (
    "context"
    "time"
)

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
```

#### `internal/usecase/telemetry/ingest.go`
```go
package telemetry

import (
    "context"
    "github.com/yourusername/telemetry-service/internal/domain/telemetry"
    "github.com/yourusername/telemetry-service/pkg/logger"
)

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
```

#### `internal/adapter/repository/timescaledb/telemetry.go`
```go
package timescaledb

import (
    "context"
    "database/sql"
    "github.com/yourusername/telemetry-service/internal/domain/telemetry"
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
```

#### `internal/adapter/http/handler/telemetry.go`
```go
package handler

import (
    "encoding/json"
    "net/http"
    "github.com/yourusername/telemetry-service/internal/usecase/telemetry"
)

type TelemetryHandler struct {
    queryUseCase *telemetry.QueryUseCase
}

func NewTelemetryHandler(queryUC *telemetry.QueryUseCase) *TelemetryHandler {
    return &TelemetryHandler{queryUseCase: queryUC}
}

func (h *TelemetryHandler) GetLatest(w http.ResponseWriter, r *http.Request) {
    vehicleID := r.URL.Query().Get("vehicle_id")
    if vehicleID == "" {
        http.Error(w, "vehicle_id is required", http.StatusBadRequest)
        return
    }
    
    telemetry, err := h.queryUseCase.GetLatest(r.Context(), vehicleID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(telemetry)
}
```

#### `cmd/server/main.go`
```go
package main

import (
    "context"
    "log"
    "os"
    "os/signal"
    "syscall"
    
    "github.com/yourusername/telemetry-service/internal/di"
    "github.com/yourusername/telemetry-service/internal/infrastructure/config"
)

func main() {
    // Load configuration
    cfg, err := config.Load()
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }
    
    // Build dependency container
    container, cleanup, err := di.BuildContainer(cfg)
    if err != nil {
        log.Fatalf("Failed to build container: %v", err)
    }
    defer cleanup()
    
    // Start services
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    // Start HTTP server
    go container.HTTPServer.Start(ctx)
    
    // Start message consumer
    go container.MessageConsumer.Start(ctx)
    
    // Wait for interrupt signal
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
    <-sigChan
    
    log.Println("Shutting down gracefully...")
}
```

#### `Makefile`
```makefile
.PHONY: help build test lint run migrate

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build the service
	go build -o bin/telemetry-service cmd/server/main.go

test: ## Run tests
	go test -v -race -coverprofile=coverage.out ./...

test-integration: ## Run integration tests
	go test -v -tags=integration ./test/integration/...

lint: ## Run linters
	golangci-lint run

run: ## Run the service locally
	go run cmd/server/main.go

migrate-up: ## Run database migrations
	migrate -path migrations -database "postgres://localhost:5432/telemetry?sslmode=disable" up

migrate-down: ## Rollback migrations
	migrate -path migrations -database "postgres://localhost:5432/telemetry?sslmode=disable" down

docker-build: ## Build Docker image
	docker build -t telemetry-service:latest -f ../../infrastructure/docker/services/telemetry-service.Dockerfile .

proto: ## Generate protobuf code
	protoc --go_out=. --go_opt=paths=source_relative \
	       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	       api/grpc/*.proto
```

---

## III. FRONTEND STRUCTURE (React + TypeScript)

```
web/dashboard/
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── assets/
│
├── src/
│   ├── app/                           # Application setup
│   │   ├── App.tsx
│   │   ├── Routes.tsx
│   │   └── providers/
│   │       ├── AuthProvider.tsx
│   │       ├── ThemeProvider.tsx
│   │       └── WebSocketProvider.tsx
│   │
│   ├── features/                      # Feature-based modules
│   │   ├── dashboard/
│   │   │   ├── components/
│   │   │   │   ├── DashboardLayout.tsx
│   │   │   │   ├── SpeedChart.tsx
│   │   │   │   ├── SteeringChart.tsx
│   │   │   │   ├── StatusCards.tsx
│   │   │   │   └── index.ts
│   │   │   ├── hooks/
│   │   │   │   ├── useTelemetry.ts
│   │   │   │   ├── useDashboardLayout.ts
│   │   │   │   └── index.ts
│   │   │   ├── services/
│   │   │   │   └── dashboardApi.ts
│   │   │   ├── store/
│   │   │   │   ├── dashboardSlice.ts
│   │   │   │   └── selectors.ts
│   │   │   ├── types/
│   │   │   │   └── dashboard.types.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── video/
│   │   │   ├── components/
│   │   │   │   ├── VideoPlayer.tsx
│   │   │   │   ├── ObjectDetectionOverlay.tsx
│   │   │   │   └── VideoControls.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useVideoStream.ts
│   │   │   │   └── useObjectDetection.ts
│   │   │   └── services/
│   │   │       └── videoStreamService.ts
│   │   │
│   │   ├── control/
│   │   │   ├── components/
│   │   │   │   ├── ControlPanel.tsx
│   │   │   │   ├── ModeSwitch.tsx
│   │   │   │   ├── EmergencyStop.tsx
│   │   │   │   └── ParameterTuning.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useControl.ts
│   │   │   │   └── useParameterTuning.ts
│   │   │   └── services/
│   │   │       └── controlApi.ts
│   │   │
│   │   ├── planning/
│   │   │   ├── components/
│   │   │   │   ├── TrajectoryVisualization.tsx
│   │   │   │   ├── PlannedVsActual.tsx
│   │   │   │   └── PlanningMetrics.tsx
│   │   │   └── hooks/
│   │   │       └── usePlanningData.ts
│   │   │
│   │   ├── replay/
│   │   │   ├── components/
│   │   │   │   ├── ReplayPlayer.tsx
│   │   │   │   ├── Timeline.tsx
│   │   │   │   └── SessionList.tsx
│   │   │   └── hooks/
│   │   │       └── useReplay.ts
│   │   │
│   │   └── auth/
│   │       ├── components/
│   │       │   ├── LoginForm.tsx
│   │       │   └── ProtectedRoute.tsx
│   │       ├── hooks/
│   │       │   └── useAuth.ts
│   │       └── services/
│   │           └── authApi.ts
│   │
│   ├── shared/                        # Shared across features
│   │   ├── components/
│   │   │   ├── ui/                   # Base UI components
│   │   │   │   ├── Button/
│   │   │   │   │   ├── Button.tsx
│   │   │   │   │   ├── Button.test.tsx
│   │   │   │   │   ├── Button.stories.tsx
│   │   │   │   │   └── Button.module.css
│   │   │   │   ├── Card/
│   │   │   │   ├── Input/
│   │   │   │   ├── Modal/
│   │   │   │   └── index.ts
│   │   │   ├── charts/               # Chart components
│   │   │   │   ├── LineChart/
│   │   │   │   ├── AreaChart/
│   │   │   │   └── index.ts
│   │   │   └── layout/
│   │   │       ├── Header.tsx
│   │   │       ├── Sidebar.tsx
│   │   │       └── Footer.tsx
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useLocalStorage.ts
│   │   │   ├── useDebounce.ts
│   │   │   └── index.ts
│   │   │
│   │   ├── utils/
│   │   │   ├── formatting.ts
│   │   │   ├── validation.ts
│   │   │   ├── dateTime.ts
│   │   │   └── index.ts
│   │   │
│   │   └── constants/
│   │       ├── routes.ts
│   │       ├── apiEndpoints.ts
│   │       └── index.ts
│   │
│   ├── core/                          # Core functionality
│   │   ├── api/
│   │   │   ├── client.ts             # Axios/Fetch client
│   │   │   ├── websocket.ts          # WebSocket client
│   │   │   └── interceptors.ts
│   │   │
│   │   ├── store/
│   │   │   ├── index.ts              # Redux store setup
│   │   │   ├── rootReducer.ts
│   │   │   └── middleware.ts
│   │   │
│   │   └── types/
│   │       ├── telemetry.types.ts
│   │       ├── control.types.ts
│   │       ├── planning.types.ts
│   │       └── index.ts
│   │
│   ├── styles/
│   │   ├── global.css
│   │   ├── variables.css
│   │   ├── themes/
│   │   │   ├── dark.css
│   │   │   └── light.css
│   │   └── mixins.css
│   │
│   ├── config/
│   │   ├── env.ts
│   │   └── app.config.ts
│   │
│   └── main.tsx                       # Application entry
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── .env.example
├── .eslintrc.js
├── .prettierrc
├── tsconfig.json
├── vite.config.ts
├── package.json
└── README.md
```

