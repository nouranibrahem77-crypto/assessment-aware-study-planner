"""Planner-facing constant mean-mark baseline predictor."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from study_planner.data.feature_engineering.constants import (
    IDENTIFIER_COLUMNS,
    STRICT_FFF_FEATURE_NAMES,
    TARGET_COLUMN,
)
from study_planner.nn.dataset import SPLIT_PRESENTATIONS

if TYPE_CHECKING:
    from study_planner.search.nn_adapter import SessionNNAdapter
    from study_planner.search.session_mapping import SessionClickMappingConfig


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SimpleAverageMarkPredictor:
    """Predict the mean of supplied training marks for every assessment row.

    This is a fitted constant baseline. Fit it only on the training partition;
    validation and test targets must never be passed to :meth:`fit`.
    """

    mean_mark: float
    identity: str = "simple-average-mark-v1"
    training_target_count: int = 0
    training_targets_sha256: str | None = None
    source_data_sha256: str | None = None
    feature_names: tuple[str, ...] = STRICT_FFF_FEATURE_NAMES
    _fitted_digest: str = field(init=False, repr=False)

    prediction_family: ClassVar[str] = "simple_average_mark"

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
        if (
            isinstance(self.training_target_count, bool)
            or not isinstance(self.training_target_count, int)
            or self.training_target_count < 0
        ):
            raise ValueError("training_target_count must be a non-negative integer")
        if tuple(self.feature_names) != STRICT_FFF_FEATURE_NAMES:
            raise ValueError("the average baseline currently requires Model A schema")
        object.__setattr__(self, "_fitted_digest", self._compute_digest())

    @property
    def assessment_type_columns(self) -> tuple[str, str, str]:
        """Return the strict feature schema's assessment type indicators."""
        return STRICT_FFF_FEATURE_NAMES[:3]

    @property
    def artifact_digest(self) -> str:
        """Compatibility name for the digest of this fitted baseline state."""
        return self._fitted_digest

    def _compute_digest(self) -> str:
        payload = {
            "identity": self.identity,
            "mean_mark": float(self.mean_mark),
            "training_target_count": self.training_target_count,
            "training_targets_sha256": self.training_targets_sha256,
            "source_data_sha256": self.source_data_sha256,
            "feature_names": self.feature_names,
            "prediction_family": self.prediction_family,
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

    def verify_integrity(self) -> str:
        """Return fitted-state digest, rejecting mutation after fitting."""
        if (
            self.training_target_count < 1
            or not self.training_targets_sha256
            or self._compute_digest() != self._fitted_digest
        ):
            raise ValueError(
                "average predictor fitted provenance is incomplete or changed"
            )
        return self._fitted_digest

    @classmethod
    def fit(
        cls,
        training_marks: Sequence[float],
        *,
        identity: str = "simple-average-mark-v1",
        source_data_sha256: str | None = None,
        feature_names: tuple[str, ...] = STRICT_FFF_FEATURE_NAMES,
    ) -> SimpleAverageMarkPredictor:
        """Fit the baseline from finite training marks in the score range."""
        if len(training_marks) == 0:
            raise ValueError("training_marks cannot be empty")
        values = np.asarray(training_marks, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("training_marks must be a finite one-dimensional sequence")
        if ((values < 0) | (values > 100)).any():
            raise ValueError("training marks must be in [0, 100]")
        return cls(
            mean_mark=float(values.mean()),
            identity=identity,
            training_target_count=len(values),
            training_targets_sha256=hashlib.sha256(
                values.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            source_data_sha256=source_data_sha256,
            feature_names=feature_names,
        )

    @classmethod
    def fit_from_processed_directory(
        cls, processed_directory: Path
    ) -> SimpleAverageMarkPredictor:
        """Fit only from the validated strict training split and its manifest."""
        manifest_path = processed_directory / "fff_preprocessing.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing strict feature manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("course_module") != "FFF"
            or manifest.get("score_feature_mode") != "strict"
            or tuple(manifest.get("feature_names", ())) != STRICT_FFF_FEATURE_NAMES
        ):
            raise ValueError(
                "average baseline requires the strict FFF feature manifest"
            )
        train_filename = manifest.get("split_files", {}).get("train")
        if (
            not isinstance(train_filename, str)
            or Path(train_filename).name != train_filename
        ):
            raise ValueError("strict feature manifest has an invalid training filename")
        train_path = processed_directory / train_filename
        if not train_path.is_file():
            raise FileNotFoundError(f"missing strict training split: {train_path}")

        required = {*IDENTIFIER_COLUMNS, TARGET_COLUMN}
        training = pd.read_csv(
            train_path,
            usecols=sorted(required),
            dtype={name: "string" for name in IDENTIFIER_COLUMNS},
        )
        allowed_presentations = SPLIT_PRESENTATIONS["train"]
        observed_presentations = set(
            training["code_presentation"].dropna().astype(str)
        )
        expected_rows = manifest.get("row_counts", {}).get("train")
        if (
            training.empty
            or set(training["code_module"].dropna().astype(str)) != {"FFF"}
            or observed_presentations != allowed_presentations
            or (expected_rows is not None and len(training) != expected_rows)
            or training.loc[:, list(IDENTIFIER_COLUMNS)].isna().any().any()
            or training.duplicated(["id_student", "id_assessment"]).any()
        ):
            raise ValueError("training split includes invalid FFF presentations")
        targets = pd.to_numeric(training[TARGET_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(targets).all() or ((targets < 0) | (targets > 100)).any():
            raise ValueError("training target marks must be finite and in [0, 100]")
        with train_path.open("rb") as source_stream:
            source_digest = hashlib.file_digest(source_stream, "sha256").hexdigest()
        return cls.fit(
            targets.tolist(),
            source_data_sha256=source_digest,
            feature_names=STRICT_FFF_FEATURE_NAMES,
        )

    def create_session_adapter(
        self, mapping_config: SessionClickMappingConfig
    ) -> SessionNNAdapter:
        """Wrap this predictor in the planner's matched-input session adapter."""
        self.verify_integrity()
        from study_planner.search.nn_adapter import SessionAverageMarkAdapter

        return SessionAverageMarkAdapter(self, mapping_config)

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
