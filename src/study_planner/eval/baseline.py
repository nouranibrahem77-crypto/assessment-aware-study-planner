"""Planner-facing constant mean-mark baseline predictor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SimpleAverageMarkPredictor:
    """Predict the mean of supplied training marks for every assessment row.

    This is a fitted constant baseline. Fit it only on the training partition;
    validation and test targets must never be passed to :meth:`fit`.
    """

    mean_mark: float
    identity: str = "simple-average-mark-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.mean_mark, bool)
            or not isinstance(self.mean_mark, int | float)
            or not math.isfinite(float(self.mean_mark))
            or not 0 <= self.mean_mark <= 100
        ):
            raise ValueError("mean_mark must be finite and in [0, 100]")
        if not self.identity:
            raise ValueError("predictor identity cannot be empty")

    @classmethod
    def fit(
        cls, training_marks: Sequence[float], *, identity: str = "simple-average-mark-v1"
    ) -> SimpleAverageMarkPredictor:
        """Fit the baseline from finite training marks in the score range."""
        if len(training_marks) == 0:
            raise ValueError("training_marks cannot be empty")
        values = np.asarray(training_marks, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("training_marks must be a finite one-dimensional sequence")
        if ((values < 0) | (values > 100)).any():
            raise ValueError("training marks must be in [0, 100]")
        return cls(float(values.mean()), identity)

    def predict(self, features: object) -> np.ndarray:
        """Return one mean-mark prediction per input row.

        ``features`` is intentionally unused; accepting a batch-shaped object
        lets this predictor share the ordinary regression prediction boundary.
        """
        try:
            row_count = len(features)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("features must be a batch with a row count") from error
        if row_count < 1:
            raise ValueError("features batch cannot be empty")
        return np.full(row_count, float(self.mean_mark), dtype=np.float64)
