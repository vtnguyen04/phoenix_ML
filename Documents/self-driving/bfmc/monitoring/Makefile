.PHONY: help build test lint run migrate

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build all services
	@echo "Building all services..."
	./scripts/build/build-all.sh

test: ## Run all tests
	@echo "Running all tests..."
	# Add test commands here

lint: ## Run linters
	@echo "Running linters..."
	# Add lint commands here

run: ## Run all services locally
	@echo "Running all services..."
	docker-compose up

migrate-up: ## Run database migrations
	@echo "Running database migrations..."
	# Add migration commands here

migrate-down: ## Rollback migrations
	@echo "Rolling back migrations..."
	# Add rollback commands here

docker-build: ## Build all Docker images
	@echo "Building all Docker images..."
	docker-compose build

proto: ## Generate protobuf code
	@echo "Generating protobuf code..."
	./scripts/build/build-proto.sh
