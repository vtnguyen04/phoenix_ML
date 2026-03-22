# ADR 005 — DVC Integration for Data Versioning

## Status

**Accepted** — Framework provides DVC integration; users configure DVC per-project.

## Context

The Phoenix ML framework supports diverse ML problem types (tabular, image, NLP, object detection). Different problem types have different data requirements:

| Problem Type | Data Format | Size | Versioning Need |
|---|---|---|---|
| Tabular (credit risk, fraud) | CSV in `data/` | Small (MB) | Low — can regenerate |
| Image Classification | NPZ/folders | Medium (GB) | Medium |
| Object Detection | Image dirs + annotations | Large (GB-TB) | **High — cannot regenerate** |
| NLP | Text corpora | Variable | High |

For large, irreplaceable datasets (images, annotations), teams need a way to **version, share, and reproduce** exact dataset versions. DVC (Data Version Control) is the industry standard for this.

## Decision

**DVC is a framework-supported integration, not a shipped configuration.**

- The framework provides **code integration** with DVC:
  - `ModelConfig.data_source_type = "dvc"` — declare DVC-backed data
  - `ModelConfig.retrain_trigger = "data_change"` — trigger retrain on DVC changes
  - `dags/data_change_pipeline.py` — Airflow DAG that monitors DVC status
  - `dvc[s3]` as a project dependency

- The framework does **NOT** ship DVC config files:
  - `.dvc/config`, `dvc.yaml`, `dvc.lock`, `.dvcignore` are **user-generated**
  - These are gitignored and set up by each team for their specific storage backend

## How Users Set Up DVC

```bash
# 1. Initialize DVC in the project
dvc init

# 2. Configure remote storage (MinIO, S3, GCS, etc.)
dvc remote add -d myremote s3://my-bucket/dvc-data
dvc remote modify myremote endpointurl http://minio:9000

# 3. Track datasets
dvc add data/object_detection/
git add data/object_detection/.gitkeep data/object_detection.dvc

# 4. Push data to remote
dvc push

# 5. Configure model to use DVC
# In model_configs/object-detection.yaml:
#   data_source:
#     type: dvc
#   retrain:
#     trigger: data_change
#     drift_detection: false
```

## Consequences

**Positive:**
- Framework stays clean — no project-specific config files
- Works with any DVC-compatible storage backend (S3, GCS, Azure, local)
- `data_change_pipeline` DAG auto-detects data changes and retrains

**Negative:**
- Users must set up DVC themselves (documented in customization guide)
- DVC dependency is always installed even if not used (acceptable tradeoff)
