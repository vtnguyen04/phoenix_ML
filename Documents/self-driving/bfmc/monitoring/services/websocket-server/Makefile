.PHONY: help build test lint run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build the service
	go build -o bin/websocket-server cmd/server/main.go

test: ## Run tests
	go test -v -race -coverprofile=coverage.out ./...

lint: ## Run linters
	golangci-lint run

run: ## Run the service locally
	go run cmd/server/main.go

docker-build: ## Build Docker image
	docker build -t websocket-server:latest -f ../../infrastructure/docker/services/websocket-server.Dockerfile .

