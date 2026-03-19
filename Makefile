.PHONY: up down logs build restart clean ps api-logs frontend-logs test

# ─── Docker Compose Commands ─────────────────────────────────────
up:                         ## Start all services
	docker compose up -d

up-build:                   ## Rebuild and start all services
	docker compose up -d --build

down:                       ## Stop all services
	docker compose down

down-clean:                 ## Stop all services and remove volumes
	docker compose down -v

restart:                    ## Restart all services
	docker compose restart

build:                      ## Build images without starting
	docker compose build

ps:                         ## Show running services
	docker compose ps

# ─── Logs ────────────────────────────────────────────────────────
logs:                       ## Follow all logs
	docker compose logs -f

api-logs:                   ## Follow API logs
	docker compose logs -f api

frontend-logs:              ## Follow frontend logs
	docker compose logs -f frontend

# ─── Development ─────────────────────────────────────────────────
test:                       ## Run tests locally
	uv run pytest tests/ -v --tb=short

lint:                       ## Run ruff + mypy
	uv run ruff check .
	uv run mypy src/ tests/ --explicit-package-bases

# ─── Utilities ───────────────────────────────────────────────────
shell:                      ## Open shell in API container
	docker compose exec api bash

db-shell:                   ## Open psql shell
	docker compose exec postgres psql -U phoenix -d phoenix_ml

redis-cli:                  ## Open redis-cli
	docker compose exec redis redis-cli

clean:                      ## Remove all containers, volumes, and images
	docker compose down -v --rmi all

help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
