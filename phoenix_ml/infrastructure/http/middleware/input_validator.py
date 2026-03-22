"""Input validation middleware — validate features against model config.

Checks:
- Feature count matches model's expected feature count
- Feature values within acceptable ranges (if configured)
- Request body size limit
"""

from __future__ import annotations

import logging
import math
from typing import Any

from phoenix_ml.application.commands.predict_command import PredictCommand

logger = logging.getLogger(__name__)


def _max_features() -> int:
    from phoenix_ml.config import get_settings  # noqa: PLC0415

    return get_settings().INPUT_MAX_FEATURES


class InputValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, error_code: str = "INVALID_INPUT"):
        super().__init__(message)
        self.error_code = error_code


def validate_prediction_input(
    command: PredictCommand,
    model_config: dict[str, Any] | None = None,
) -> list[str]:
    """Validate prediction input against model config.

    Returns list of validation errors (empty = valid).
    """
    errors: list[str] = []

    # 1. Basic checks
    if not command.features:
        errors.append("Features list cannot be empty")
        return errors

    max_f = _max_features()
    if len(command.features) > max_f:
        errors.append(
            f"Too many features: {len(command.features)} "
            f"(max: {max_f})"
        )

    # 2. Model config validation
    if model_config:
        expected_count = model_config.get("feature_count")
        if expected_count and len(command.features) != expected_count:
            errors.append(
                f"Feature count mismatch: got {len(command.features)}, "
                f"expected {expected_count} for model {command.model_id}"
            )

        feature_names = model_config.get("feature_names", [])
        if feature_names and len(command.features) != len(feature_names):
            errors.append(
                f"Feature count doesn't match model schema: "
                f"got {len(command.features)}, "
                f"schema has {len(feature_names)} features"
            )

        # Range validation
        ranges = model_config.get("feature_ranges", {})
        for i, (name, (lo, hi)) in enumerate(ranges.items()):
            if i < len(command.features):
                val = command.features[i]
                if val < lo or val > hi:
                    errors.append(
                        f"Feature '{name}' value {val} "
                        f"out of range [{lo}, {hi}]"
                    )

    # 3. NaN / Inf check
    for i, val in enumerate(command.features):
        if math.isnan(val):
            errors.append(f"Feature at index {i} is NaN")
        elif math.isinf(val):
            errors.append(f"Feature at index {i} is infinite")

    if errors:
        logger.warning(
            "Input validation failed for model %s: %s",
            command.model_id,
            errors,
        )

    return errors
