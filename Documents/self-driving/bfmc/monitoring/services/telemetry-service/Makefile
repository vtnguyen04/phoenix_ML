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
