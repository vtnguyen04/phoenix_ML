# Contributing to Phoenix ML Platform

Thank you for your interest in contributing! This guide helps you get set up.

## Development Setup

```bash
# Clone
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML

# Install with all dependencies (requires uv)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Code Standards

### Architecture
The project follows **Domain-Driven Design (DDD)** with **Clean Architecture**:

```
src/
├── domain/          # Business logic, entities, interfaces (no external deps)
├── application/     # Use-case handlers, DTOs, commands
├── infrastructure/  # Concrete implementations (DB, HTTP, ML engines)
└── shared/          # Cross-cutting (exceptions, utils)
```

### Rules
- **SOLID principles** must be followed
- All code must pass: `ruff check .`, `mypy . --explicit-package-bases`, `pytest`
- Domain layer must **NOT** import from infrastructure
- All public functions require type hints and docstrings
- New files must have `__init__.py` with proper exports
- No hardcoded model names or ports — use `os.environ.get()` with defaults

### Formatting
```bash
uv run ruff check .       # Lint (195 files)
uv run ruff check --fix . # Auto-fix
uv run ruff format .      # Format
```

## Adding a New ML Example

1. Create `examples/<problem_name>/train.py`
2. Implement `train_and_export(output_path, metrics_path, reference_path)` function
3. Create `model_configs/<problem-name>.yaml`:

```yaml
model_id: my-model
version: v1
framework: onnx
task_type: classification  # or regression
model_path: models/my_model/v1/model.onnx
train_script: examples/my_model/train.py
dataset_name: my-dataset
feature_names: [f1, f2, f3]
metadata:
  role: champion
```

4. Add stage in `dvc.yaml`
5. Run: `uv run python examples/<problem_name>/train.py`

## Testing

```bash
uv run pytest tests/unit/         # Unit tests
uv run pytest tests/integration/  # Integration tests
uv run pytest --cov=src           # With coverage

# Load test (requires locust)
uv run locust -f benchmarks/load_test.py --host http://localhost:8001
```

## Running Locally

```bash
# Start API server (port 8000)
uv run uvicorn src.infrastructure.http.fastapi_server:app --reload

# Or via CLI entry point (after pip install -e .)
phoenix-serve

# Full stack (Docker — port 8001)
docker compose up -d --build
docker compose -f docker-compose.airflow.yaml up -d
```

## Pull Request Checklist

- [ ] `ruff check .` passes
- [ ] `mypy . --explicit-package-bases` passes
- [ ] All tests pass: `pytest`
- [ ] New code has tests
- [ ] `__init__.py` updated with exports
- [ ] No hardcoded values (model names, ports, paths)
- [ ] Documentation updated if applicable
