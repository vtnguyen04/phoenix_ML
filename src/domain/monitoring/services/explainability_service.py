"""Model explainability via permutation-based feature importance.

Note: SHAP is incompatible with Python 3.13. We use a permutation-based
approach that works with any ONNX model and doesn't require SHAP.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Compute feature importance for a single prediction.

    Method: perturbation-based local importance.
    For each feature, perturb it (set to 0) and measure output change.
    Larger change → more important feature.
    """

    async def explain(
        self,
        engine: Any,
        model: Any,
        features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute per-feature importance for a prediction.

        Args:
            engine: InferenceEngine instance
            model: Model entity
            features: 1-D float32 array of feature values
            feature_names: Optional names for each feature

        Returns:
            dict with 'prediction', 'feature_importances', 'top_features'
        """
        from src.domain.inference.value_objects.feature_vector import (  # noqa: PLC0415
            FeatureVector,
        )

        features = features.astype(np.float32).flatten()
        n_features = len(features)
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        # Baseline prediction
        baseline_pred = await engine.predict(model, FeatureVector(values=features))
        baseline_val = float(baseline_pred.result)

        # Perturb each feature and measure change
        importances: dict[str, float] = {}
        for i in range(n_features):
            perturbed = features.copy()
            perturbed[i] = 0.0  # Zero-out this feature
            pert_pred = await engine.predict(model, FeatureVector(values=perturbed))
            pert_val = float(pert_pred.result)
            importances[names[i]] = abs(baseline_val - pert_val)

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: round(v / total, 4) for k, v in importances.items()}

        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return {
            "prediction": baseline_val,
            "confidence": float(baseline_pred.confidence.value),
            "importances": dict(sorted_features),
            "top_features": [f for f, _ in sorted_features[:5]],
            "method": "perturbation",
        }
