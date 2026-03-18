# ADR 005: DVC for Data and Model Versioning

## Status
Accepted

## Context
ML projects require reproducible pipelines where data, models, and code are versioned together. Git alone cannot handle large binary files (ONNX models, datasets). We need a system that:
- Tracks data/model artifacts alongside code
- Stores large files in external storage (S3/MinIO)
- Defines reproducible training pipelines
- Integrates with Git workflow

## Decision
We adopted **DVC (Data Version Control)** with **MinIO** as the S3-compatible remote storage.

### Pipeline Stages (`dvc.yaml`)
1. **train-champion**: Train GradientBoostingClassifier → `models/credit_risk/v1/model.onnx`
2. **train-challenger**: Train LogisticRegression → `models/credit_risk/v2/model.onnx`
3. **seed-features**: Generate reference distributions → `data/reference_data.json`

### Storage
- Remote: MinIO at `s3://dvc` with auto-created bucket via `createbuckets` service
- Lock file: `dvc.lock` tracks exact hashes for reproducibility

## Consequences

### Positive
- **Reproducibility**: `dvc repro` recreates exact training artifacts
- **Collaboration**: Team members run `dvc pull` to get artifacts without retraining
- **CI integration**: `dvc.lock` in Git enables automated pipeline validation
- **Cost-effective**: MinIO runs locally; swap to AWS S3 for production

### Negative
- **Extra dependency**: `dvc[s3]` adds ~30 transitive packages
- **Learning curve**: Team must understand DVC commands alongside Git
