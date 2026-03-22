"""Model explainability via SHAP and perturbation-based feature importance.

Supports:
- SHAP TreeExplainer (for tree-based models: XGBoost, LightGBM, Random Forest)
- SHAP KernelExplainer (for any model, slower)
- Perturbation-based (fallback for ONNX models without SHAP support)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try importing SHAP (available on Python 3.12)
try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.info("SHAP not available, using perturbation-based explainability")


class ExplainabilityService:
    """Compute feature importance for predictions.

    Methods (in priority order):
    1. SHAP KernelExplainer — model-agnostic, accurate  (if shap installed)
    2. Perturbation-based — fallback, always works
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

        # Try SHAP if available
        if _HAS_SHAP:
            try:
                return await self._explain_shap(
                    engine, model, features, names, baseline_val, baseline_pred
                )
            except Exception:
                logger.warning("SHAP explain failed, falling back to perturbation", exc_info=True)

        # Fallback: perturbation-based
        return await self._explain_perturbation(
            engine, model, features, names, baseline_val, baseline_pred
        )

    async def _explain_shap(
        self,
        engine: Any,
        model: Any,
        features: np.ndarray,
        names: list[str],
        baseline_val: float,
        baseline_pred: Any,
    ) -> dict[str, Any]:
        """SHAP KernelExplainer — model-agnostic SHAP values."""
        from src.domain.inference.value_objects.feature_vector import (  # noqa: PLC0415
            FeatureVector,
        )

        def predict_fn(x: np.ndarray) -> np.ndarray:
            """Synchronous wrapper for async engine.predict."""
            import asyncio  # noqa: PLC0415

            results = []
            loop = asyncio.new_event_loop()
            try:
                for row in x:
                    pred = loop.run_until_complete(
                        engine.predict(model, FeatureVector(values=row.astype(np.float32)))
                    )
                    results.append(float(pred.result))
            finally:
                loop.close()
            return np.array(results)

        # Create background data (variations around the input)
        background = features.reshape(1, -1) + np.random.randn(10, len(features)) * 0.1  # noqa: NPY002

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(features.reshape(1, -1), nsamples=50)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_abs = np.abs(shap_values.flatten())
        total = float(shap_abs.sum())
        importances: dict[str, float] = {}
        for i, name in enumerate(names):
            importances[name] = round(float(shap_abs[i]) / total, 4) if total > 0 else 0.0

        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return {
            "prediction": baseline_val,
            "confidence": float(baseline_pred.confidence.value),
            "importances": dict(sorted_features),
            "top_features": [f for f, _ in sorted_features[:5]],
            "method": "shap_kernel",
        }

    async def _explain_perturbation(
        self,
        engine: Any,
        model: Any,
        features: np.ndarray,
        names: list[str],
        baseline_val: float,
        baseline_pred: Any,
    ) -> dict[str, Any]:
        """Perturbation-based: zero out each feature and measure impact."""
        from src.domain.inference.value_objects.feature_vector import (  # noqa: PLC0415
            FeatureVector,
        )

        importances: dict[str, float] = {}
        for i, name in enumerate(names):
            perturbed = features.copy()
            perturbed[i] = 0.0
            pert_pred = await engine.predict(model, FeatureVector(values=perturbed))
            importances[name] = abs(baseline_val - float(pert_pred.result))

        total = sum(importances.values())
        if total > 0:
            importances = {k: round(v / total, 4) for k, v in importances.items()}

        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return {
            "prediction": baseline_val,
            "confidence": float(baseline_pred.confidence.value),
            "importances": dict(sorted_features),
            "top_features": [f for f, _ in sorted_features[:5]],
            "method": "perturbation",
        }
