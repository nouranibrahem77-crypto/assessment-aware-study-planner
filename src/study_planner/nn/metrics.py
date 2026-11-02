"""Regression metrics and the shared score-bound policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREDICTION_MIN = 0.0
PREDICTION_MAX = 100.0
TARGET_SCALE = 100.0


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    """Apply the explicit [0, 100] score output bound."""
    return np.clip(
        np.asarray(predictions, dtype=np.float64), PREDICTION_MIN, PREDICTION_MAX
    )


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    assessment_types: Sequence[object] | np.ndarray,
) -> dict[str, Any]:
    """Calculate clipped overall and per-assessment-type regression metrics."""
    y_true = np.asarray(targets, dtype=np.float64)
    y_pred = clip_predictions(predictions)
    types = np.asarray(assessment_types, dtype=object)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or types.shape != y_true.shape:
        raise ValueError("Targets, predictions, and types must be aligned vectors")

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        actual = y_true[mask]
        estimated = y_pred[mask]
        if not len(actual):
            return {
                "n": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
            }
        return {
            "n": int(len(actual)),
            "mae": float(mean_absolute_error(actual, estimated)),
            "rmse": float(np.sqrt(mean_squared_error(actual, estimated))),
            "r2": float(r2_score(actual, estimated))
            if len(actual) > 1
            else float("nan"),
        }

    output: dict[str, Any] = {"overall": summarize(np.ones(len(y_true), dtype=bool))}
    output["by_assessment_type"] = {
        str(category): summarize(types == category)
        for category in sorted(set(types.tolist()), key=str)
    }
    return output


def metric_json_safe(metrics: Any) -> Any:
    """Replace undefined metric values with JSON null recursively."""
    if isinstance(metrics, dict):
        return {key: metric_json_safe(value) for key, value in metrics.items()}
    if isinstance(metrics, list):
        return [metric_json_safe(value) for value in metrics]
    if isinstance(metrics, float) and not np.isfinite(metrics):
        return None
    return metrics
