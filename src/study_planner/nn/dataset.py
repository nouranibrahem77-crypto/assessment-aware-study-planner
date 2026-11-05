"""Strict OULAD feature loading and model-input schema validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from study_planner.data.feature_engineering.constants import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)

SPLIT_PRESENTATIONS: dict[str, frozenset[str]] = {
    "train": frozenset({"2013B", "2013J"}),
    "validation": frozenset({"2014B"}),
    "test": frozenset({"2014J"}),
}
ASSESSMENT_TYPE_PREFIX = "assessment_type_"


class DatasetSchemaError(ValueError):
    """Raised when processed features violate the saved strict schema."""


@dataclass(frozen=True)
class FeatureSplit:
    """Model inputs, target, and audit-only metadata for one split."""

    X: pd.DataFrame
    y: np.ndarray
    assessment_types: np.ndarray
    identifiers: pd.DataFrame


@dataclass(frozen=True)
class StrictFeatureData:
    """Validated requested rows from strict feature split files."""

    feature_names: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    assessment_type_columns: tuple[str, ...]
    splits: dict[str, FeatureSplit]
    manifest: dict[str, Any]


def _model_frame(
    raw: pd.DataFrame,
    feature_names: tuple[str, ...],
    assessment_type_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = sorted(set(feature_names) - set(raw.columns))
    extra = sorted(set(raw.columns) - set(feature_names))
    if missing or extra:
        raise DatasetSchemaError(
            f"Input feature schema mismatch; missing={missing}, extra={extra}"
        )
    if any(name.startswith("previous_assessment_score") for name in feature_names):
        raise DatasetSchemaError("Strict inputs cannot include previous score features")

    raw_types = raw.loc[:, list(assessment_type_columns)]
    type_values = raw_types.apply(pd.to_numeric, errors="coerce")
    if (raw_types.notna() & type_values.isna()).any().any():
        raise DatasetSchemaError("Assessment type indicators must be numeric")
    if ((type_values.notna()) & ~type_values.isin([0, 1])).any().any():
        raise DatasetSchemaError("Assessment type indicators must be 0 or 1")
    missing_indicators = type_values.isna().any(axis=1)
    active_count = type_values.fillna(0).sum(axis=1)
    invalid = (~missing_indicators) & (active_count != 1)
    if invalid.any():
        raise DatasetSchemaError("Each row must have exactly one assessment type")

    type_names = [
        name.removeprefix(ASSESSMENT_TYPE_PREFIX).upper()
        for name in assessment_type_columns
    ]
    category = type_values.fillna(0).to_numpy().argmax(axis=1)
    categories = np.asarray(type_names, dtype=object)[category]
    categories[missing_indicators.to_numpy()] = np.nan

    numeric_names = tuple(
        name for name in feature_names if name not in assessment_type_columns
    )
    raw_numeric = raw.loc[:, list(numeric_names)]
    numeric = raw_numeric.apply(pd.to_numeric, errors="coerce")
    if (raw_numeric.notna() & numeric.isna()).any().any():
        raise DatasetSchemaError("Numeric input features must contain numbers")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    if np.isinf(numeric_values).any():
        raise DatasetSchemaError("Numeric input features cannot contain infinity")
    frame = numeric.copy()
    frame.insert(0, "assessment_type", pd.Series(categories, index=raw.index))
    return frame, categories


def load_strict_feature_data(
    processed_dir: Path,
    split_names: tuple[str, ...] = ("train", "validation", "test"),
) -> StrictFeatureData:
    """Load requested strict FFF splits without touching unrequested files.

    Training workflows that must not access held-out test data should request
    only ``("train", "validation")``. Both splits are required for model fit
    and selection; test is optional.
    """
    if len(split_names) != len(set(split_names)):
        raise DatasetSchemaError("Requested split names must be unique")
    unknown_splits = set(split_names) - set(SPLIT_PRESENTATIONS)
    if unknown_splits:
        raise DatasetSchemaError(
            f"Unknown requested split(s): {sorted(unknown_splits)}"
        )
    if not {"train", "validation"}.issubset(split_names):
        raise DatasetSchemaError("Strict model data requires train and validation")
    manifest_path = processed_dir / "fff_preprocessing.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("course_module") != "FFF":
        raise DatasetSchemaError("Expected FFF feature manifest")
    if manifest.get("score_feature_mode") != "strict":
        raise DatasetSchemaError("Primary model training requires strict feature mode")
    if manifest.get("target") != TARGET_COLUMN:
        raise DatasetSchemaError("Manifest target does not match target_score")
    feature_names = tuple(manifest.get("feature_names", ()))
    if not feature_names:
        raise DatasetSchemaError("Manifest has no feature names")
    if len(feature_names) != len(set(feature_names)):
        raise DatasetSchemaError("Manifest contains duplicate feature names")
    forbidden = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN}
    if forbidden.intersection(feature_names):
        raise DatasetSchemaError("Identifiers or target listed as model features")
    assessment_type_columns = tuple(
        name for name in feature_names if name.startswith(ASSESSMENT_TYPE_PREFIX)
    )
    if not assessment_type_columns:
        raise DatasetSchemaError("No assessment type feature columns in manifest")

    split_files = manifest.get("split_files", {})
    splits: dict[str, FeatureSplit] = {}
    for split_name in split_names:
        expected_presentations = SPLIT_PRESENTATIONS[split_name]
        filename = split_files.get(split_name)
        if not isinstance(filename, str):
            raise DatasetSchemaError(f"Manifest lacks {split_name} split file")
        path = processed_dir / filename
        raw = pd.read_csv(
            path,
            dtype={name: "string" for name in IDENTIFIER_COLUMNS},
            keep_default_na=True,
        )
        expected_row_count = manifest.get("row_counts", {}).get(split_name)
        if expected_row_count is not None and len(raw) != expected_row_count:
            raise DatasetSchemaError(
                f"{filename} has {len(raw)} rows; manifest expects {expected_row_count}"
            )
        required_columns = {*IDENTIFIER_COLUMNS, *feature_names, TARGET_COLUMN}
        missing = sorted(required_columns - set(raw.columns))
        extra = sorted(set(raw.columns) - required_columns)
        if missing or extra:
            raise DatasetSchemaError(
                f"{filename} columns mismatch; missing={missing}, extra={extra}"
            )
        if raw.loc[:, list(IDENTIFIER_COLUMNS)].isna().any().any():
            raise DatasetSchemaError(f"{filename} has missing audit identifiers")
        if set(raw["code_module"].astype(str)) != {"FFF"}:
            raise DatasetSchemaError(f"{filename} contains a non-FFF course row")
        observed = set(raw["code_presentation"].dropna().astype(str))
        if observed != expected_presentations:
            raise DatasetSchemaError(
                f"Unexpected {split_name} presentations: {sorted(observed)}"
            )
        if raw[TARGET_COLUMN].isna().any():
            raise DatasetSchemaError(f"{filename} has missing target values")
        target = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(target).all() or ((target < 0) | (target > 100)).any():
            raise DatasetSchemaError("Target scores must be finite and in [0, 100]")

        model_inputs, categories = _model_frame(
            raw.loc[:, list(feature_names)], feature_names, assessment_type_columns
        )
        if any(name in model_inputs.columns for name in forbidden):
            raise DatasetSchemaError("Identifiers or target entered model inputs")
        splits[split_name] = FeatureSplit(
            X=model_inputs,
            y=target,
            assessment_types=categories,
            identifiers=raw.loc[:, list(IDENTIFIER_COLUMNS)].copy(),
        )

    numeric_feature_names = tuple(
        name for name in feature_names if name not in assessment_type_columns
    )
    return StrictFeatureData(
        feature_names=feature_names,
        numeric_feature_names=numeric_feature_names,
        assessment_type_columns=assessment_type_columns,
        splits=splits,
        manifest=manifest,
    )


def prepare_inference_frame(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
    assessment_type_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Validate a feature-only batch and map it to the fitted model schema."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Inference features must be a pandas DataFrame")
    forbidden = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN}
    if forbidden.intersection(features.columns):
        raise DatasetSchemaError("Inference batch contains identifiers or target")
    missing = sorted(set(feature_names) - set(features.columns))
    extra = sorted(set(features.columns) - set(feature_names))
    if missing or extra:
        raise DatasetSchemaError(
            f"Inference feature schema mismatch; missing={missing}, extra={extra}"
        )
    ordered = features.reindex(columns=feature_names)
    return _model_frame(ordered, feature_names, assessment_type_columns)[0]
